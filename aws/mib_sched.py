import numpy as np
from collections import deque
import itertools
from toolz.functoolz import compose_left
import pandas as pd
import networkx as nx
import logging
from inst_set import df_insert_row
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from utils import omega_rwc_bitwidth
import sys
sys.path.append('./figure')
from micro57_pset import blue_code, green_code, yellow_code
import matplotlib.gridspec as gridspec

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def contains_number(list_of_numbers, target_number):
	""" Function to check if a number appears in the list"""
	return target_number in list_of_numbers

def find_empty_slots(bit_map_after_merge):
	return np.where(np.any(bit_map_after_merge, axis=1) == False)[0]

class BaseSched:
	def __init__(self, iscaC, pdimMax, SkipO3, plotName, CtrlOnly=False): 
		self.iscaC=iscaC
		self.pipeStage = np.log2(iscaC).astype(int)
		""" U280 Total II delay 
			AWS 64 delay 51
			U50 64 delay 60
		"""
		pipeIIdelay={16: 33, 32: 38, 64: 43}
		assert iscaC in pipeIIdelay
		""" Instruction Scheduling Delay for True Dependency """
		self.pipeDelay = pipeIIdelay.get(iscaC)
		""" Instruction Scheduling Delay for Anti Dependency """
		pipeAntiDelay={16: 33, 32: 38, 64: 43}
		self.antiDelay = pipeAntiDelay.get(iscaC)
		""" For scheduling"""
		self.rrIndices=[]
		self.addrsRemit=[]
		""" For code generation """
		self.mulsREmit=[]
		self.addrWemit=[]
		""" Instruction Dependency Graph """
		self.ddNxG = nx.DiGraph()
		""" Max Address Space used by the network program """
		self.heightRF = pdimMax//self.iscaC
		""" TODO: CHECK nullAddr below scaler register files offset on CVB """
		self.nullAddr = 2*self.heightRF # pay attention not to overwrite rigister file
		""" Network instruction bit width of each field """
		ctrlBW, readBW, writeBW = omega_rwc_bitwidth(self.pipeStage)
		assert readBW == writeBW
		self.capacityOfRF = 2**readBW
		self.readBitShift = ctrlBW
		self.writeBitShift = ctrlBW + readBW
		""" Skip Out Of Order Compression """
		self.SkipO3 = SkipO3
		""" Intermediate plot for debugging """
		self.plotName = plotName
		""" Only return the control signal """
		self.CtrlOnly=CtrlOnly

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

	def log_stream(self, recordIn):
		print(recordIn)
		return recordIn

	def base_sched(self, rrTable):
		""" Don't do compression at all """
		instCount = rrTable.shape[0]
		instSched = np.ones(instCount, dtype=np.uint32)*-1
		instSched[0] = 0
		for instItem in range(1, instCount):
			instSched[instItem] = instSched[instItem-1] + self.pipeDelay
		assert (instSched>=0).all()
		return instSched

	def sched_constrain(self):
		""" Build Resource Table """
		rrTable = np.zeros(
			(len(self.rrIndices), self.iscaC*(self.pipeStage+1)), 
			dtype=bool) 

		routeCtrl = np.zeros(
			(len(self.rrIndices), self.iscaC), 
			dtype=np.uint32) 

		""" Used for Resource Reservation Table visualization, 
			uint8 for color the working model of each node in the network """
		rrImg = np.zeros(
			(len(self.rrIndices), self.iscaC*(self.pipeStage+1)), 
			dtype=np.uint8) 

		for idx, (readRFs, writeRF) in enumerate(self.rrIndices):
			""" Set input readRFs """
			rrTable[idx, readRFs] = True
			rrImg[idx, readRFs] = 1
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

				""" Image showing node work modes"""
				rrImg[idx, self.iscaC*(stage_item+1):self.iscaC*(stage_item+2):] = nodeSignal
	
		return rrTable, routeCtrl, rrImg

	def list_sched(self, rrTable):
		instCount = rrTable.shape[0]
		resourceWidth = rrTable.shape[1]
		instSched = np.ones(instCount, dtype=np.int32)*-1
		""" Leave redundancy due to pipeline delay """
		condense_bitmap = np.zeros(
			(instCount, resourceWidth), 
			dtype=bool) 

		for instItem in range(instCount):
			"""	Assert all parent nodes are scheduled,
				Find earliest slot that satisfy constraint """
			InstParents = list(self.ddNxG.predecessors(instItem))
			if len(InstParents) > 0:
				assert all(x >= 0 for x in instSched[InstParents]),\
					print('Parent not scheduled yet:', 
							instSched[InstParents], InstParents)
				incomingEdges = self.ddNxG.in_edges(instItem, data='weight')
				edgeDelay = [weight for _, _, weight in incomingEdges]
				earliestSlot = max(instSched[InstParents]+edgeDelay)
			else:
				earliestSlot = 0
			""" Test the fit of all possible locations """
			bit_map_single = rrTable[instItem,:]
			""" Grow bitmap if schedule overflow """
			schedGrow=earliestSlot-condense_bitmap.shape[0]

			if schedGrow>=0:
				condense_bitmap = np.concatenate(
					(condense_bitmap, 
	  				np.zeros((schedGrow+1, resourceWidth), dtype=bool)),
					axis=0)

			bit_map_after_merge = np.logical_and(
				condense_bitmap[earliestSlot:,:], 
				bit_map_single)

			""" Search the first fit"""
			EmptySlots = find_empty_slots(bit_map_after_merge)
			if len(EmptySlots)<1:
				""" grow if all full """
				condense_bitmap = np.concatenate(
								(condense_bitmap, 
								np.zeros((1, resourceWidth), dtype=bool)),
								axis=0)
				firstFitSlot = bit_map_after_merge.shape[0] + earliestSlot
			else:
				firstFitSlot = EmptySlots[0] + earliestSlot

			instSched[instItem] = firstFitSlot
			condense_temp_slice = condense_bitmap[firstFitSlot,:] 
			""" Update the merged row """
			condense_bitmap[firstFitSlot,:] = np.logical_or(bit_map_single, condense_temp_slice)

		assert (instSched>=0).all()
		return instSched

	def hbm_fill(self, instSched, routeCtrl):
		zipHeight = max(instSched) + 1
		logging.debug(f"{self.plotName:10s} O3 {zipHeight:5d}/{len(instSched):5d}") 

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
		if self.CtrlOnly:
			return hbmInst.view(np.float32)
		else:
			""" horizonal concate """
			return np.hstack((hbmMul, hbmInst.view(np.float32)))

	def code_gen(self):
		"""	Form the graph and table constraints """
		rrTable, routeCtrl, self.rrImg = self.sched_constrain()

		if self.SkipO3:
			self.instSched = self.base_sched(rrTable)
		else:
			self.instSched = self.list_sched(rrTable)

		hbmMul = self.hbm_fill(self.instSched, routeCtrl)
		return hbmMul.flatten()

	def rrTable_plot(self, ax, rrImg):
		markerShape=','
		for (idx, colorItem) in enumerate([green_code, yellow_code, blue_code]): 
			ax.spy(rrImg==idx+1,  
				marker=markerShape,
				markeredgecolor=colorItem,
				markerfacecolor=colorItem)
		ax.set_xticks([rrImg.shape[1]])
		ax.set_yticks([rrImg.shape[0]])
	
	def plot_o3sched(self):
		""" Illustrate the Out-Of-Order schedule and 
			compression of the instruction flow """
		rrImg=self.rrImg.T

		zipHeight = max(self.instSched) + 1
		rrZip = np.zeros(
			(self.iscaC*(self.pipeStage+1), zipHeight), 
			dtype=np.uint8) 
		for (idx, zItem) in enumerate(self.instSched):
			rrZip[:, zItem] += rrImg[:, idx]

		widthPixel = rrImg.shape[1]
		heightPixel = rrImg.shape[0]*2
		heightInch=5
		widthInch=heightInch*widthPixel/heightPixel

		upper_width = 1.0
		lower_width = rrZip.shape[1]*1.0/widthPixel

		""" Align left of the O3 before and after """
		fig = plt.figure(figsize=(widthInch, heightInch))
		gs = gridspec.GridSpec(2, 2, 
						 height_ratios=[1, 1], 
						 width_ratios=[upper_width, 1 - lower_width])

		ax1 = fig.add_subplot(gs[0, :])
		self.rrTable_plot(ax1, rrImg)
		ax1.set_title('Before O3')

		ax2 = fig.add_subplot(gs[1, 0])
		self.rrTable_plot(ax2, rrZip)
		ax2.set_title('After O3')

		plt.tight_layout()
		plt.savefig('./temp/AfterO3-'+self.plotName+'.pdf', transparent=True)

	def plot_dependency(self):
		plt.figure()
		copyG = self.ddNxG.copy()
		isolated_nodes = [node for node, degree in dict(copyG.degree()).items() if degree == 0]
		copyG.remove_nodes_from(isolated_nodes)

		logging.debug("Dependency Graph %d nodes %d edges", 
				copyG.number_of_nodes(),
				copyG.number_of_edges())

		graphLayout = nx.spring_layout(copyG, k=0.8)
		# graphLayout = graphviz_layout(copyG, "twopi")
		# graphLayout = graphviz_layout(copyG, "circo")
		# graphLayout = graphviz_layout(copyG, "fdp")
		# graphLayout = graphviz_layout(copyG, "sfdp")

		# Scale to avoid overlap
		layoutScale = 1.0 
		graphLayout= {node: (x + layoutScale * i, y + layoutScale * i) 
				for i, (node, (x, y)) in enumerate(graphLayout.items())}

		""" Draw labels without circle 
		nx.draw_networkx_labels(copyG, 
						  	graphLayout,
		  					font_size = 6)
		nx.draw_networkx_edges(copyG, 
						  	graphLayout,
							edge_color='gray')
		"""

		""" Draw edge labels 
		edgeLabels = nx.get_edge_attributes(copyG, 'weight')
		nx.draw_networkx_edge_labels(
			copyG, 
			graphLayout, 
		  	font_size = 6,
			edge_labels=edgeLabels)
		"""

		""" Simplified drawing, no node and edge label """
		# Set node sizes based on their degree
		# nodeSizes = [copyG.degree(node) * 10 for node in copyG.nodes()]
		nx.draw(copyG, 
		#   graphLayout,
		#   node_size = nodeSizes,
		#   node_size = 0.5,
		  node_color='black', 
		  edge_color='gray', 
		  )

		plt.axis('off')
		plt.savefig('./temp/Dependency-'+self.plotName+'.pdf', transparent=True)

		""" Export the ddDF """
		self.ddDf.to_csv('./temp/ddDf.csv')

class ShuffleMulSched(BaseSched):
	def __init__(self, iscaC,
			  pdimMax,
			  srcOrder,
			  dstOrder,
			  muls,
			  SkipO3=False,
			  plotName='fig'): 
		""" O3 schedule for permutation """
		super().__init__(iscaC=iscaC, 
				   pdimMax=pdimMax, 
				   SkipO3=SkipO3,
				   plotName=plotName)
		self.srcOrder=srcOrder
		self.dstOrder=dstOrder
		self.muls=muls

	def top_pass(self):
		s_in = self.pair_src_dst()
		rr_pass = compose_left(
			enumerate,
			lambda x: map(self.build_inst_dependency, x),
		)

		""" Apply pass """
		list(rr_pass(s_in))

		return self.code_gen()

	def pair_src_dst(self):
		for srcItem, dstItem, mulItem in zip(self.srcOrder, self.dstOrder, self.muls):
			readRF, addrR = self.addr_1d_2d(srcItem)
			writeRF, addrW = self.addr_1d_2d(dstItem)
			yield (readRF, mulItem, addrR, writeRF, addrW)

	def build_inst_dependency(self, recordIn):
		""" Dependancy Graph, recordIn:
		0: instID
		1-0: readRF
		1-1: mulR
		1-2: addrR
		1-3: writeRF
		1-4: addrW """
		instID = recordIn[0]
		readRF = recordIn[1][0]
		mulR = recordIn[1][1]
		addrR = recordIn[1][2].astype(np.uint32)
		writeRF = recordIn[1][3]
		addrW = recordIn[1][4]
		""" No dependency at all, no edges in the graph """
		self.ddNxG.add_node(instID)

		""" Used for build resource reservation table """
		self.rrIndices.append(([readRF], writeRF))
		self.mulsREmit.append(mulR)
		self.addrsRemit.append(addrR)
		self.addrWemit.append(addrW)

class MatMulSched(BaseSched):
	""" O3 Instruction Scheduling Based on Dragon Book Chapter 10 """
	def __init__(self, 
			  SpMatrix, 
			  iscaC, 
			  readOffset,
			  pdimMax,
			  plotName='fig',
			  SkipO3=False,
			  CtrlOnly=False):
		super().__init__(iscaC=iscaC, 
				   pdimMax=pdimMax, 
				   SkipO3=SkipO3,
				   plotName=plotName,
				   CtrlOnly=CtrlOnly)
		self.mat = SpMatrix
		self.readOffset = readOffset
		self.poolEnd = self.iscaC-1

		""" Each idxTerm has seperate psum pool"""
		self.psumPool = 2*self.heightRF + 1 
		self.TermStack = [] 

		""" Instruction Property Table """
		self.ddDf = pd.DataFrame(
			columns=['Terminal', 'tempWrite', 'Read'])

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
			# lambda x: map(self.log_stream,x), # DEBUG
			lambda x: map(self.psum_dependency, x),
		)
		""" Apply pass """
		list(rr_pass(s_in))

		return self.code_gen()

	def pStack_ptr(self, idxTerm):
		if idxTerm not in self.TermStack:
			self.TermStack.append(idxTerm)
		return self.psumPool + self.TermStack.index(idxTerm)

	def true_dependency(self, idxTerm, prevResult, instID):
		def match_RbW(df):
			matching_indices = df[(df['Terminal'] == idxTerm) & (df['tempWrite'] == prevResult)].index.tolist()
			return matching_indices
		matches = match_RbW(self.ddDf)
		if len(matches)>0:
			self.ddNxG.add_edge(matches[-1], instID, 
					   weight=self.pipeDelay) 

	def anti_dependency(self, currWrite, instID):
		def match_WbR(df):
			df = self.ddDf['Read'].apply(lambda x: contains_number(x, currWrite))
			matching_indices = df[(df == True)].index.tolist()
			return matching_indices
		matches = match_WbR(self.ddDf)
		if len(matches)>0:
			self.ddNxG.add_edge(matches[-1], instID,
					   weight=self.antiDelay) 

	def plot_pattern(self):
		plt.figure()
		plt.spy(self.mat, 
			 markersize=1, 
			marker='.',
			markeredgecolor='black',
			markerfacecolor='black')
		plt.xticks([],[])
		plt.yticks([],[])
		plt.title('Sparse Pattern')
		plt.savefig('./temp/Pattern-'+self.plotName+'.png')

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

	def psum_dependency(self, recordIn):
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

		""" Solve true dependency (Read before Write, RbW) and
			anti dependency (Write before Read, WbR) """
		currWrite = self.addr_2d_1d(writeRF, addrW)
		self.ddNxG.add_node(instID)

		""" Add edges for RbW """
		indicesR = self.addr_2d_1d(readRFs, addrsR)
		for prevResult in indicesR:
			self.true_dependency(idxTerm, prevResult, instID)

		""" Look for WbR of current result """
		self.anti_dependency(currWrite, instID)

		""" Add current write for later true dependency """
		df_insert_row(self.ddDf, [idxTerm, currWrite, indicesR])

		""" Used for build resource reservation table """
		self.rrIndices.append((readRFs, writeRF))
		self.mulsREmit.append(mulsR)
		self.addrsRemit.append(addrsR)
		self.addrWemit.append(addrW)

class LsolveSched(MatMulSched):
	def __init__(self, 
			  iscaC, 
			  pdimMax, 
			  SpMatrix,
			  SkipO3=False,
			  plotName='fig'):
		super().__init__(
			SpMatrix=SpMatrix,
			iscaC=iscaC, 
			readOffset=0,
			pdimMax=pdimMax,
			SkipO3=SkipO3,
			plotName=plotName)

	def top_pass(self):
		s_in = self.row_stream()
		""" Transform to the rr table with instruction dependancy tree """
		rr_pass = compose_left(
			lambda x: map(self.RRFs_assign, x),
			itertools.chain.from_iterable,
			lambda x: map(self.Psum_reduce, x),
			itertools.chain.from_iterable,
			enumerate,
			lambda x: map(self.psum_dependency, x),
		)
		""" Apply pass """
		list(rr_pass(s_in))

		""" Add Lower solve dependency """
		df = self.ddDf
		termDf = df[df['Terminal'] == df['tempWrite']]
		termList=np.ones(self.mat.shape[0]).astype(np.int32)*-1
		termList[termDf['Terminal'].tolist()]=termDf.index.tolist()

		for instID, row in self.ddDf.iterrows():
			termItem = row['Terminal']
			readList = row['Read'].tolist()
			""" Remove self dependency """
			if termItem in readList:
				readList.remove(termItem)
			""" Remove psum temp write dependency """
			readList = [x for x in readList if x<len(termList)]
			""" Remove empty row in L dependency """
			SrcList = [x for x in termList[readList] if x >=0]
			""" Add Edge if not empty"""
			if bool(SrcList):	
				EdgeList = [(x, instID) for x in SrcList]
				self.ddNxG.add_edges_from(EdgeList,weight=self.pipeDelay)

		return self.code_gen()

	def row_stream(self):
		for idxTerm in range(self.mat.shape[0]):
			row_start = self.mat.indptr[idxTerm]
			row_end = self.mat.indptr[idxTerm+1]
			if row_start == row_end:
				continue
			yield (idxTerm, 
				np.append(self.mat.indices[row_start:row_end], idxTerm),
				np.append(-self.mat.data[row_start:row_end], 1.0))

class UsolveSched(LsolveSched):
	def __init__(self, 
			  iscaC, 
			  pdimMax, 
			  SpMatrix,
			  SkipO3=False,
			  plotName='fig'): 
		super().__init__(
			SpMatrix=SpMatrix,
			iscaC=iscaC, 
			pdimMax=pdimMax,
			SkipO3=SkipO3,
			plotName=plotName)

	def row_stream(self):
		for idxTerm in reversed(range(self.mat.shape[0])):
			row_start = self.mat.indptr[idxTerm]
			row_end = self.mat.indptr[idxTerm+1]
			if row_start == row_end:
				continue
			yield (idxTerm, 
				np.append(self.mat.indices[row_start:row_end], idxTerm),
				np.append(-self.mat.data[row_start:row_end], 1.0))

class UpFactorSched(LsolveSched):
	def __init__(self, 
			  iscaC, 
			  pdimMax, 
			  SpMatrix,
			  etree,
			  Lnz,
			  SkipO3=False):
		super().__init__(
			SpMatrix=SpMatrix,
			iscaC=iscaC, 
			pdimMax=pdimMax,
			SkipO3=SkipO3)
		self.etree=etree
		self.Lnz=Lnz

	def top_pass(self):
		s_in = self.row_stream()
		rr_pass = compose_left(
			itertools.chain.from_iterable,
		)
		""" Apply pass """
		list(rr_pass(s_in))

		# return self.code_gen()

	def row_stream(self):
		""" numeric factor QDLDL_factor() 
			in the qdldl.c, inputs: """
		n=self.mat.shape[0]
		Ap = self.mat.indptr
		Ai = self.mat.indices
		Ax = self.mat.data
		etree = self.etree
		Lnz = self.Lnz
		""" Solve a series of y = L(0:(k-1),0:k-1)) \ b """	
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
		yVals = np.zeros(n, dtype=np.float32)
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

			""" Symbolic phase, etree Reach """
			nnzY = 0
			for i in range(col_start, col_end):
				bidx = Ai[i]
				nextIdx = bidx
				if yMarkers[nextIdx] == QDLDL_UNUSED:
					yMarkers[nextIdx] = QDLDL_USED

					elimBuffer[0] = nextIdx
					nnzE = 1

					nextIdx = etree[bidx]# walk through etree
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

			""" Numeric Solve y = L \ b 
				Through column elimination """
			for i in reversed(range(0, nnzY)):
				cidx = yIdx[i] # column index
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

		print(Lx)
		return Lp, Li, Lx, D
