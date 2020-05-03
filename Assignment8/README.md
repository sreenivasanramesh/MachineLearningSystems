## Assignment 8 Dense Matrix Multiplication Comparison: Handcoded, Breeze, Spark

### Files to submit:
A report in PDF format, no more than 2 pages.  
Handcoded Scala code for executing 1000x1000 dense matrix multiplication (leftMatrix 1000x1000, rightMatrix 1000x1000) 
for N times and get the average time for executing one matrix multiplication. Spark code for executing 
1000x1000 local dense matrix multiplication for N times and get the average time for executing one matrix multiplication. 
Scala code using Breeze for executing 1000x1000 local dense matrix multiplication for N times and get the average time 
for executing one matrix multiplication. N can be set depending on your platform to control that the total 
time of running N times is around 3 minutes. On my platform, N is chosen as 100. If you don't know, just choose N = 1000.  


### Learning Goal:
- Understand and apply Matrix Multiplication Implementation on Spark (local mode) and Breeze;
- Understand the performance implication of different matrix multiplication implementations;
- Practice scala programming;
- Learn the setup and installation of Spark and Breeze


### Tasks:

 - Implement three programs:  
    1. create two 1000x1000 random matrices, write a function to use multiple loops to multiply these two matrices, write an outer loop to execute the matrix multiplication by 1000 times; run the program, and measure the matrix multiplication computation time.
    2. Perform above task on Spark. You can write two 1000x1000 random matrices to text files, and load these into Spark. You can also create the random matrices in Spark. Use the DenseMatrix class introduced in the class.
    3. Perform above task using Breeze.
    4. Write a report to summarize the performance comparison results:
        - list your testing machine hardware configuration: number of cores,  core type, core frequency, size of memory, and so on;
        - list your measured average time for one matrix multiplication for each of the three implementations. 
        - summarize what you have observed and learnt from the performance comparison.

### Grading Criteria:
- Handcoded implementation can work: 3pt
- Spark implementation can work: 3pt
- Breeze implementation can work: 3pt
- In the report, machine hardware configuration is listed: 3pt
- In the report,  measured time for the three implementations are listed and compared in a way that is easy to understand: 4pt
- In the report, observations and things learnt from the implementation and performance comparison are summarized: 4pt

