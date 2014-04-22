1. run the code:
   in Main.m, execute each code section respectively

2. image and accuracy data are generated in /data/output folder

3. face dataset should be put in /data/AgeFaceDataset/ folder
 

a brief introduction of the project code structure is give:

-- data
    -- AgeFaceDataSet  
    -- output
    -- train.txt  
    -- test.txt    
-- src
	-- ReadData.m
    -- DisplayFace.m
	-- NFM.m
    -- PCA.m
	-- LDA.m
	-- GMM.m
	-- Main.m


-- ReadData.m 
   auxilary function for reading data. It can read data into data matrix.
	
-- DisplayFace.m 
   display feature vector as face \& rescale pixel value to 0-255. 

-- NFM.m 
   learning Nonegative Matrix Factorization with 50 bases (bases can be reset,
   support random initialization)
		
-- PCA.m 
   do PCA decompostion and reconstruction, support choosing number of PCs
		
-- LDA.m 
   do LDA basis finding
		
-- GMM.m 
   finding GMM Model using EM algorithm
		
-- Main.m
   All subtasks are organized into sections 

