1. run the code:
   in Main.m, execute each code section respectively

2. image and accuracy data are generated in /data/output folder

3. face dataset should be put in /data/AgeFaceDataset/ folder
 

a brief introduction of the project code structure is give:
\begin{itemize}
\item[--] data
	\begin{itemize}
	  \item[--] AgeFaceDataSet  
	  \item[--] output
	  \item[--] train.txt  
      \item[--] test.txt  
	\end{itemize}  
\item[--] src
	\begin{itemize}
		\item[--] ReadData.m
		\item[--] DisplayFace.m
		%\item[--] PlotResult.m
		\item[--] NFM.m
		\item[--] PCA.m
		\item[--] LDA.m
		\item[--] GMM.m
		\item[--] Main.m
  	\end{itemize}
\end{itemize}


\subsection{code explaination}

\begin{itemize}
	\item[--] ReadData.m \\
		auxilary function for reading data. It can read data into data matrix.
	
	\item[--] DisplayFace.m \\
		display feature vector as face \& rescale pixel value to 0-255. 
	%\item[--] PlotResult.m
	\item[--] NFM.m \\
		learning Nonegative Matrix Factorization with 50 bases (bases can be reset,
		support random initialization)
		
	\item[--] PCA.m \\
		do PCA decompostion and reconstruction, support choosing number of PCs
		
	\item[--] LDA.m \\
		do LDA basis finding
		
	\item[--] GMM.m \\
		finding GMM Model using EM algorithm
		
	\item[--] Main.m\\
		All subtasks are organized into sections \\
		\begin{enumerate}
			\item PCA: 
			PCA.m to get basis and DisplayFace.m show eigenface \\
			PCA to features and KNN on these features, plot accuracy
				
			\item NMF:
			NFM.m to get basis and DisplayFace.m to show faces
			
			\item LDA:
			LDA.m to get Fisherfaces and DisplayFace.m to show faces \\
			LDA(can also use PCA before LDA once) to project feature and KNN on these
			features, plot accuracy(compare with PCA only)
			
			\item GMM:
			GMM.m(can use PCA before GMM) to get 8 centres and DisplayFace.m to show
			faces
			
		\end{enumerate}
\end{itemize}