import numpy as np
from collections import deque
import itertools
from toolz.functoolz import compose_left
import pandas as pd
import networkx as nx
import sys
sys.path.append('./graph')
import logging
from inst_set import df_insert_row
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from utils import omega_rwc_bitwidth

class DragonC10:
	""" O3 Instruction Scheduling Based on Dragon Book Chapter 10 """
	def __init__(self, 
			  csr_mat, 
			  iscaC, 
			  readOffset,
			  pdimMax
			  ):
		self.mat = csr_mat
		self.iscaC = iscaC
		self.readOffset = readOffset
		self.heightRF = pdimMax//self.iscaC
		self.poolEnd = self.iscaC-1
		""" Each idxTerm has seperate psum pool"""
		self.nullAddr = 2*self.heightRF 
		self.psumPool = 2*self.heightRF + 1 
		self.TermStack = [] 

		self.pipeStage = np.log2(iscaC).astype(int)
		""" U280 Total II delay """
		pipeIIdelay={16:43, 32: 51, 64:51}
		assert iscaC in pipeIIdelay
		self.pipeDelay = pipeIIdelay.get(iscaC)

		self.rrIndices=[]
		self.mulsREmit=[]
		self.addrsRemit=[]
		self.addrWemit=[]
		""" Instruction Dependency Graph """
		self.ddNxG = nx.DiGraph()
		""" Instruction Property Table """
		self.ddDf = pd.DataFrame(
			columns=['Terminal', 'tempWrite'])

		ctrlBW, readBW, writeBW, _ = omega_rwc_bitwidth(self.pipeStage)
		assert readBW == writeBW
		self.capacityOfRF = 2**readBW
		self.readBitShift = ctrlBW
		self.writeBitShift = ctrlBW + readBW

	def pStack_ptr(self, idxTerm):
		if idxTerm not in self.TermStack:
			self.TermStack.append(idxTerm)
		return self.psumPool + self.TermStack.index(idxTerm)

	def match_node(self, idxTerm, prevWrite):
		df = self.ddDf
		matching_indices = df[(df['Terminal'] == idxTerm) & (df['tempWrite'] == prevWrite)].index.tolist()
		return matching_indices

	def depend_src(self, idxTerm, currWrite, instID):
		""" Write op: Source of data dependency 
			Used NX and DF for Analysis, Convert to Dot for visualize """
		self.ddNxG.add_node(instID)
		df_insert_row(self.ddDf, [idxTerm, currWrite])

	def depend_dst(self, idxTerm, prevWrite, instID,):
		""" Read op: Destination of data dependency """
		matches = self.match_node(idxTerm, prevWrite)
		if len(matches)>0:
			# print((idxTerm, prevWrite), 'match', matches,(idxTerm, instID))
			self.ddNxG.add_edge(matches[-1], instID) 
			#    weight=self.pipeStage)

	def addr_1d_2d(self, addr_1d):
		bank_num = addr_1d % self.iscaC
		bank_loc = addr_1d//self.iscaC
		return bank_num, bank_loc

	def addr_2d_1d(self, bank_num, bank_loc):
		return bank_num + (bank_loc * self.iscaC)

	def IdxToBin(self, idx):
		""" Turn the binary representation of an int32 number to a byte array """
		cvt_temp = np.array([idx]).view(np.uint8)
		return np.unpackbits(cvt_temp, bitorder = 'little')[:self.pipeStage]

	def draw_dep_sched(self, instSched):
		""" Plot Graph """
		copyG = self.ddNxG.copy()
		isolated_nodes = [node for node, degree in dict(copyG.degree()).items() if degree == 0]
		copyG.remove_nodes_from(isolated_nodes)

		# graphLayout = nx.spring_layout(copyG, k=0.8)
		# graphLayout = graphviz_layout(copyG, "twopi")
		# graphLayout = graphviz_layout(copyG, "circo")
		# graphLayout = graphviz_layout(copyG, "fdp")
		graphLayout = graphviz_layout(copyG, "sfdp")

		nx.draw(copyG, 
		  graphLayout,
		  with_labels=True, 
		  node_color='skyblue', 
		  edge_color='gray', 
		  alpha=0.7)
		plt.savefig('./temp/ddNxG.png')

		""" Plot the schedule with Stage Graph """
		"""
		slotMap = [np.where(instSched == slot)[0] 
			 for slot in range(max(instSched)+1)]
		print('instSched', instSched)
		for idx, sItem in enumerate(slotMap):
			print(sItem)
			if idx >10:
				break
		"""
		# plt.savefig('./temp/instSched.png')

	def top_pass(self):
		# s_in = itertools.islice(self.row_stream(), 20)
		s_in = self.row_stream()
		""" Transform to the rr table with instruction dependancy tree """
		rr_pass = compose_left(
			lambda x: map(self.RRFs_assign, x),
			itertools.chain.from_iterable,
			lambda x: map(self.Psum_reduce, x),
			itertools.chain.from_iterable,
			enumerate,
			lambda x: map(self.ddDotG_rrTable_build, x),
		)
		""" Apply pass """
		list(rr_pass(s_in))

		"""	Form the graph and table constraints """
		rrTable, routeCtrl = self.sched_constrain()

		""" List schedule in the Dragon book """
		instSched = self.list_sched(rrTable)

		""" Draw the schedule result """
		# self.draw_dep_sched(instSched)

		hbmMul = self.code_gen(instSched, routeCtrl)

		return hbmMul.flatten()

	def sched_constrain(self):
		""" Build Resource Table """
		rrTable = np.zeros(
			(len(self.rrIndices), self.iscaC*(self.pipeStage+1)), 
			dtype=bool) 

		routeCtrl = np.zeros(
			(len(self.rrIndices), self.iscaC), 
			dtype=np.uint32) 

		for idx, (readRFs, writeRF) in enumerate(self.rrIndices):
			""" Set input readRFs """
			rrTable[idx, readRFs] = True
			dstBin = self.IdxToBin(writeRF)
			srcBins = [self.IdxToBin(x) for x in readRFs]
			flipFlags = [np.bitwise_xor(x, dstBin) for x in srcBins]
			workNodes = readRFs
			""" Merge nodes for next stage """
			_, mergeFilter = np.unique(workNodes, return_index=True)
			for stage_item in range(self.pipeStage):
				flipBits = [x[stage_item] for x in flipFlags]
				""" Flip the readRF idx encoding stage by stage """
				workNodes = workNodes ^ (np.array(flipBits)<<stage_item)
				rrTable[idx, workNodes+self.iscaC*(stage_item+1)] = True
				""" Set node signal """
				crossNodes = workNodes[np.array(flipBits, dtype=bool)]
				""" Ignore previously merged nodes"""
				mergedNodes=workNodes[mergeFilter]
				binCount = np.bincount(mergedNodes, minlength=self.iscaC)	
				""" Cross encode as 2 """
				binCount[crossNodes] += 1
				assert (binCount<=3).all()
				nodeSignal = binCount.astype(np.uint32)
				""" Shift """
				routeCtrl[idx, :] += (nodeSignal<<(2*stage_item))

				""" Merge nodes for next stage """
				_, mergeFilter = np.unique(workNodes, return_index=True)
	
		return rrTable, routeCtrl

	def code_gen(self, instSched, routeCtrl):
		zipHeight = max(instSched) + 1
		logging.debug("compressed %d origin %d", 
				zipHeight,
				len(instSched))

		""" Codegen, no inter loop conflict, can be parallelized """
		hbmMul = np.zeros((zipHeight, self.iscaC), dtype=np.float32)
		hbmInst = np.zeros((zipHeight, self.iscaC), dtype=np.uint32)
		hbmInst += (self.nullAddr<<self.writeBitShift)
		for rIdx, schedPtr in enumerate(instSched):
			(readRFs, writeRF) = self.rrIndices[rIdx]
			hbmMul[schedPtr, readRFs] = self.mulsREmit[rIdx]
			""" Set Bits """
			hbmInst[schedPtr, :] += routeCtrl[rIdx] 
			hbmInst[schedPtr, readRFs] += (self.addrsRemit[rIdx]<<self.readBitShift) 
			""" Clear nullAddr"""
			hbmInst[schedPtr, writeRF] -= (self.nullAddr<<self.writeBitShift)
			hbmInst[schedPtr, writeRF] += (self.addrWemit[rIdx]<<self.writeBitShift) 
			""" Assert read and write RF address within range """
			assert self.addrWemit[rIdx] < self.capacityOfRF
			assert (self.addrsRemit[rIdx] < self.capacityOfRF).all()

		""" horizonal concate """
		# return np.concatenate((hbmMul, hbmInst.view(np.float32)), axis=1)
		return np.hstack((hbmMul, hbmInst.view(np.float32)))

	def row_stream(self):
		for idxTerm in range(self.mat.shape[0]):
			row_start = self.mat.indptr[idxTerm]
			row_end = self.mat.indptr[idxTerm+1]
			if row_start == row_end:
				continue
			yield (idxTerm, 
		  		self.mat.indices[row_start:row_end]+self.readOffset,
				self.mat.data[row_start:row_end])

	def RRFs_assign(self, recordIn):
		""" Resolve R_RF conflict, recordIn: 
		0: idxTerm
		1: indicesR
		2: mulsR """
		idxTerm, indicesR, mulsR = recordIn 
		readRFs, addrsR = self.addr_1d_2d(indicesR)
		dq_array = []
		for _ in range(self.iscaC):
			dq_array.append(deque())
		for RF_item, Addr_item, Mul_item in zip(readRFs, addrsR, mulsR):
			dq_array[RF_item].append((Addr_item, Mul_item))

		def queues_height():
			queue_heights = list(map(lambda q: len(q), dq_array))
			return max(queue_heights)

		max_height = queues_height()
		while True:
			currHeight = queues_height()
			""" Exit when all values are marked packed """
			if currHeight == 0:
				break

			""" Get one pack of unique bank access """
			RF_emit, Mul_emit, Addr_emit = [], [], []
			for RF_item, q_item in enumerate(dq_array):
				if bool(q_item):
					Addr_item, Mul_item = q_item.popleft()
					RF_emit.append(RF_item)
					Mul_emit.append(Mul_item)
					Addr_emit.append(Addr_item)
			RF_emit = np.array(RF_emit)
			Mul_emit = np.array(Mul_emit)
			Addr_emit = np.array(Addr_emit)
			""" cur_height = 1 means last partial sum """
			recordOut = (RF_emit, # 0
							Mul_emit, # 1
							Addr_emit, # 2
							idxTerm, # 3
							currHeight, # 4
							max_height-currHeight, # 5
							)
			yield recordOut

	def Psum_reduce(self, recordIn):
		""" Break Psums, recordIn:
		0: readRFs
		1: mulsR
		2: addrsR
		3: idxTerm
		4: currHeight
		5: partialCnt """
		idxTerm = recordIn[3]
		currHeight = recordIn[4]
		partialCnt = recordIn[5]
		poolPtr = partialCnt % self.poolEnd

		def case_dst_final():
			""" Single Psum"""
			recordTemp3, recordTemp4 =\
				  self.addr_1d_2d(idxTerm)
			recordOut = (recordIn[0],
				 			recordIn[1],
				 			recordIn[2],
							recordTemp3, # 3
							recordTemp4# 4
							)
			return recordOut

		def case_psum_item(stackPtr):
			""" Intermediate Psums """
			recordOut = (recordIn[0],
				 			recordIn[1],
				 			recordIn[2],
							poolPtr, # 3
							stackPtr # 4
							)
			return recordOut

		def case_psum_reduce(partialCnt, writeRF, addrW, stackPtr):
			""" Combine Psums when pool or train end """
			psumRFs = np.arange(poolPtr+1).astype(np.int32)	
			psumMuls = np.ones(poolPtr+1).astype(np.float32)	
			psumAddrs = np.full(poolPtr+1, stackPtr).astype(np.int32)
			if partialCnt != poolPtr:
				""" Include psum from last round """
				psumRFs=np.append(psumRFs, self.poolEnd)
				psumMuls=np.append(psumMuls, 1.0)
				psumAddrs=np.append(psumAddrs, stackPtr)

			recordOut = (psumRFs,
				 			psumMuls,
				 			psumAddrs,
							writeRF, # 3
							addrW # 4
							)
			return recordOut

		if currHeight == 1:
			if partialCnt == 0:
				""" No Dependency """
				yield case_dst_final() + (idxTerm,)
			else:
				""" Depend on all sums in the pool """
				stackPtr = self.pStack_ptr(idxTerm)
				yield case_psum_item(stackPtr)+ (idxTerm,)
				recordTemp3, recordTemp4 =\
					  self.addr_1d_2d(idxTerm)
				yield case_psum_reduce(partialCnt,
						   				recordTemp3, 
						   				recordTemp4, 
										stackPtr) + (idxTerm,)
		else:
			stackPtr = self.pStack_ptr(idxTerm)
			yield case_psum_item(stackPtr) + (idxTerm,)
			if poolPtr == self.poolEnd-1:
				""" Depend on all sums in the pool """
				yield case_psum_reduce(partialCnt, 
						   				self.poolEnd, 
						   				stackPtr,
										stackPtr) + (idxTerm,)

	def ddDotG_rrTable_build(self, recordIn):
		""" Dependancy Graph, recordIn:
		0: instID
		1-0: readRFs
		1-1: mulsR
		1-2: addrsR
		1-3: writeRF
		1-4: addrW 
		1-5: idxTerm """
		instID = recordIn[0]
		readRFs = recordIn[1][0]
		mulsR = recordIn[1][1]
		addrsR = recordIn[1][2].astype(np.uint32)
		writeRF = recordIn[1][3]
		addrW = recordIn[1][4]
		idxTerm = recordIn[1][5]

		currWrite = self.addr_2d_1d(writeRF, addrW)
		""" Check idxTerms and add Nodes """
		self.depend_src(idxTerm, currWrite, instID)

		""" check indicesR and add Edges """
		indicesR = self.addr_2d_1d(readRFs, addrsR)
		for prevWrite in indicesR:
			self.depend_dst(idxTerm, prevWrite, instID)

		""" Used for build resource reservation table """
		self.rrIndices.append((readRFs, writeRF))
		self.mulsREmit.append(mulsR)
		self.addrsRemit.append(addrsR)
		self.addrWemit.append(addrW)

	def list_sched(self, rrTable):
		instCount = rrTable.shape[0]
		resourceWidth = rrTable.shape[1]
		instSched = np.ones(instCount, dtype=np.uint32)*-1
		""" Leave redundancy due to pipeline delay """
		condense_bitmap = np.zeros(
			(instCount+self.pipeDelay*2, resourceWidth), 
			dtype=bool) 

		for instItem in range(instCount):
			"""	Assert all parent nodes are scheduled,
				Find earliest slot that satisfy constraint """
			InstParents = list(self.ddNxG.predecessors(instItem))
			if len(InstParents) > 0:
				assert all(x >= 0 for x in instSched[InstParents]),\
					print('Parent not scheduled yet:', 
							instSched[InstParents])
				earliestSlot = max(instSched[InstParents])+ self.pipeDelay
			else:
				earliestSlot = 0
			""" Test the fit of all possible locations """
			bit_map_single = rrTable[instItem,:]
			bit_map_after_merge = np.logical_and(
				condense_bitmap[earliestSlot:,:], 
				bit_map_single)
			""" Select the first fit"""
			firstFitSlot = np.where(np.any(bit_map_after_merge, axis=1) == False)[0][0] + earliestSlot
			instSched[instItem] = firstFitSlot
			condense_temp_slice = condense_bitmap[firstFitSlot,:] 
			""" Update the merged row """
			condense_bitmap[firstFitSlot,:] = np.logical_or(bit_map_single, condense_temp_slice)
		assert (instSched>=0).all()
		return instSched
