package com.disml
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.exp

object MatrixMultiplyBreeze{
	def main(args: Array[String]){

		val runs = 1000
		val n = 1000 //for test use smaller n=2
		val array_one = DenseMatrix.rand[Double](1000, 1000)
		val array_two = DenseMatrix.rand[Double](1000, 1000)

		var result = array_one * array_two
		//uncomment to verify
                //print(array_one)
		//print(attay_two)
		//println(result)

		val start = System.currentTimeMillis()
		for (_ <- 1 to runs)
			result = array_one * array_two
		val end = System.currentTimeMillis()

		println(s"\nTime for $runs runs: ${(end-start)/1000.0} s\n")
		
	}

}

