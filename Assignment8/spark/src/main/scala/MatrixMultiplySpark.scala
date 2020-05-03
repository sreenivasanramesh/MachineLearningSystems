package com.disml
import org.apache.spark.mllib.linalg.DenseMatrix

object MatrixMultiplySpark{
	def main(args: Array[String]){

		val random = new scala.util.Random //java.security.SecureRandom
		def random2dArray(dim1: Int, dim2: Int, maxValue: Int): DenseMatrix= {
			new DenseMatrix(dim1, dim2, Array.fill(dim1 * dim2)(random.nextInt(maxValue)))
		}
		
		val runs = 1000
		val n = 1000 //for test use smaller n=2
		val array_one = random2dArray(n, n, 100)
		val array_two = random2dArray(n, n, 100)

		//uncomment this to verify results
		//println(array_one)
		//println(array_two)
		var result = array_one.multiply(array_two)
		//println(result)

		val start = System.currentTimeMillis()
		for (_ <- 1 to runs)
			result = array_one.multiply(array_two)
		val end = System.currentTimeMillis()

		println(s"\nTime for $runs runs: ${(end-start)/1000.0} s\n")
		
	}

}

