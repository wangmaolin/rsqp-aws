import scipy.sparse as spa
import numpy as np
import qdldl

def perm_vec2mat(amd_order):
	""" Create a permutation matrix """
	n = len(amd_order)
	rows = np.arange(n)
	cols = amd_order 
	data = np.ones(n)
	perm_mat = spa.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
	return perm_mat

def QDLDL_etree(n, Ap, Ai):
	""" QDLDL_etree in qdldl.c """
	QDLDL_UNKNOWN = -1
	work = np.zeros(n, dtype=np.int32)
	Lnz = np.zeros(n, dtype=np.int32)
	etree = np.full(n, QDLDL_UNKNOWN, dtype=np.int32)

	for j in range(n):
		work[j] = j
		for p in range(Ap[j], Ap[j+1]):
			i = Ai[p]
			assert i<=j, print(i, j)
			while work[i] != j:
				if etree[i] == QDLDL_UNKNOWN:
					etree[i] = j
				Lnz[i] += 1
				work[i] = j
				i = etree[i]

	return etree, Lnz

def QDLDL_factor(n, Ap, Ai, Ax, etree, Lnz):
	""" QDLDL_factor() in the qdldl.c
	Solve a series of y = L(0:(k-1),0:k-1)) \ b """	
	sumLnz = np.sum(Lnz)
	Dinv = np.zeros(n)
	yIdx = np.zeros(n, dtype=np.int32)
	elimBuffer = np.zeros(n, dtype=np.int32)
	LNextSpaceInCol = np.zeros(n, dtype=np.int32)
	Lp = np.zeros(n+1, dtype=np.int32)
	Li = np.zeros(sumLnz, dtype=np.int32)
	Lx = np.zeros(sumLnz, dtype=np.float32)
	QDLDL_UNUSED = 0
	QDLDL_USED = 1
	QDLDL_UNKNOWN = -1

	yMarkers = np.full(n, QDLDL_UNUSED, dtype=np.int32)
	yVals = np.zeros(n)
	D = np.zeros(n)
	for i in range(n):
		Lp[i+1] = Lp[i] + Lnz[i]
		LNextSpaceInCol[i] = Lp[i]

	D[0] = Ax[0]
	Dinv[0] = 1/D[0]

	for k in range(1, n):
		col_start = Ap[k]
		col_end = Ap[k+1] - 1

		""" Initialise y(bidx) = b(bidx) """
		diag_idx = Ai[col_end]
		assert diag_idx == k
		D[k] = Ax[col_end]
		b_indices = Ai[col_start:col_end]
		yVals[b_indices] = Ax[col_start:col_end]

		nnzY = 0
		for i in range(col_start, col_end):
			""" etree Reach, can be done offline """
			bidx = Ai[i]
			nextIdx = bidx
			if yMarkers[nextIdx] == QDLDL_UNUSED:
				yMarkers[nextIdx] = QDLDL_USED

				elimBuffer[0] = nextIdx
				nnzE = 1

				nextIdx = etree[bidx]
				while nextIdx != QDLDL_UNKNOWN and nextIdx < k:
					# Mark visited node
					if yMarkers[nextIdx] == QDLDL_USED:
						break
					yMarkers[nextIdx] = QDLDL_USED
					# Record node
					elimBuffer[nnzE] = nextIdx
					nnzE += 1
					# Move on to next node
					nextIdx = etree[nextIdx]

				# put the elimination list in the reverse order
				yIdx[nnzY:nnzY+nnzE] = elimBuffer[:nnzE][::-1]
				nnzY += nnzE

		""" Solve y = L \ b through column elimination """
		for i in reversed(range(0, nnzY)):
			cidx = yIdx[i]
			tmpIdx = LNextSpaceInCol[cidx]
			yVals_cidx = yVals[cidx]
			for j in range(Lp[cidx], tmpIdx):
				yVals[Li[j]] -= Lx[j]*yVals_cidx

			Li[tmpIdx] = k
			Lx[tmpIdx] = yVals_cidx * Dinv[cidx]
			D[k] -= yVals_cidx * Lx[tmpIdx]
			LNextSpaceInCol[cidx] += 1

			yVals[cidx] = 0.0
			yMarkers[cidx] = QDLDL_UNUSED

		Dinv[k] = 1/D[k]

	return Lp, Li, Lx, D

def symbolic_factor(csc_P, csc_A, sigma, rho):
	""" Form the KKT matrix and factor it """
	kkt_mat = spa.bmat([[csc_P+sigma*spa.identity(csc_P.shape[0]), csc_A.T], 
						[csc_A, (-1.0/rho)*spa.identity(csc_A.shape[0])]], 
						format='csc')
	gt_solver = qdldl.Solver(kkt_mat)
	gt_csc_L, diag_D, amd_order = gt_solver.factors()

	""" QDLDL_factor() in the qdldl.c"""	
	amd_mat = perm_vec2mat(amd_order)
	perm_KKT = spa.triu(amd_mat @ kkt_mat @ amd_mat.T, format='csc')

	etree, Lnz = QDLDL_etree(
		n = perm_KKT.shape[0],
		Ap = perm_KKT.indptr,
		Ai = perm_KKT.indices)

	factor_dict = {}
	factor_dict['csc_L'] = gt_csc_L
	factor_dict['diag_D'] = diag_D
	factor_dict['amd_order'] = amd_order
	factor_dict['kkt_mat'] = kkt_mat
	factor_dict['etree'] = etree
	factor_dict['Lnz'] = Lnz
	factor_dict['perm_KKT'] = perm_KKT

	return factor_dict

def qdldl_verify(csc_P, csc_A, sigma, rho):
	factor_dict = symbolic_factor(csc_P, csc_A, sigma, rho)
	gt_csc_L = factor_dict['csc_L']
	perm_KKT = factor_dict['perm_KKT ']

	Lp, Li, Lx, D = QDLDL_factor(
		n = perm_KKT.shape[0],
		Ap = perm_KKT.indptr,
		Ai = perm_KKT.indices,
		Ax = perm_KKT.data,
		etree = factor_dict['etree'],
		Lnz = factor_dict['Lnz'])

	# csc_L = spa.csc_matrix((Lx, Li, Lp), shape=(perm_KKT.shape[0], perm_KKT.shape[0]))

	""" Verification """
	np.testing.assert_array_almost_equal(gt_csc_L.indices, Li)
	np.testing.assert_array_almost_equal(gt_csc_L.indptr, Lp)
	np.testing.assert_array_almost_equal(gt_csc_L.data, Lx)
	np.testing.assert_array_almost_equal(factor_dict['diag_D'], D)
