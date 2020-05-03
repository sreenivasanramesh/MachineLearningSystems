scalaVersion := "2.11.12"

name := "matrix-multiply-spark"
version := "1.0"
libraryDependencies ++= Seq(
    "com.github.fommil.netlib" % "all" % "1.1.2",
    "org.apache.spark" %% "spark-core" % "2.4.5",
    "org.apache.spark" %% "spark-mllib" % "2.4.5"
)

assemblyMergeStrategy in assembly := {
  case PathList("aopalliance-1.0.jar", xs @ _*) => MergeStrategy.last
  case PathList("aopalliance-repackaged-2.4.0-b34.jar", xs @ _*) => MergeStrategy.last
  case PathList("arrow-format-0.10.0.jar", xs @ _*) => MergeStrategy.last
  case PathList("arrow-memory-0.10.0.jar", xs @ _*) => MergeStrategy.last
  case PathList("arrow-vector-0.10.0.jar", xs @ _*) => MergeStrategy.last
  case PathList("commons-beanutils-1.7.0.jar", xs @ _*) => MergeStrategy.last
  case PathList("commons-collections-3.2.2.jar", xs @ _*) => MergeStrategy.last
  case PathList("hadoop-yarn-api-2.6.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("hadoop-yarn-common-2.6.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("javax.inject-1.jar", xs @ _*) => MergeStrategy.last
  case PathList("javax.inject-2.4.0-b34.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-catalyst_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-core_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-graphx_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-kvstore_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-launcher_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-mllib-local_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-mllib_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-network-common_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-network-shuffle_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-sketch_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-sql_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-streaming_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-tags_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("spark-unsafe_2.12-2.4.5.jar", xs @ _*) => MergeStrategy.last
  case PathList("unused-1.0.0.jar", xs @ _*) => MergeStrategy.last
  case PathList("META-INF", _@_*) => MergeStrategy.discard
  case _ => MergeStrategy.first
}

