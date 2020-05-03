
object MatrixMultiply{
	def main(args: Array[String]): Unit = {

		val random = new scala.util.Random //java.security.SecureRandom
		def random2dArray(dim1: Int, dim2: Int, maxValue: Int) = Array.fill(dim1, dim2){
			1 + random.nextInt(maxValue)
		}
		
		val runs = 1000
		val n = 1000 //test with smaller n=2
		val array_one = random2dArray(n, n, 100)
		val array_two = random2dArray(n, n, 100)
		var result = Array.ofDim[Int](n, n)


		def multiply(array_one: Array[Array[Int]], array_two: Array[Array[Int]]) : Unit = {
			
			for (i <- 0 to array_one.length-1)
				for (j <- 0 to array_two(0).length-1)
					for (k <- 0 to array_two.length-1)
						result(i)(j) = result(i)(j) + array_one(i)(k) * array_two(k)(j)
		}
	
		//multiply(array_one, array_two)
		//println(result.map(_.mkString("  ")).mkString("\n"))

		val start = System.currentTimeMillis()
		for (_ <- 1 to runs)
			multiply(array_one, array_two)
		val end = System.currentTimeMillis()

		println(s"\nTime for $runs runs: ${(end-start)/1000.0} s \n")

	}

}

