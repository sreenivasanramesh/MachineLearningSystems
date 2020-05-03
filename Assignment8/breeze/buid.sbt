scalaVersion := "2.11.12"

name := "matrix-multiply-breeze"
version := "1.0"
libraryDependencies ++= Seq(
    "org.scalanlp" %% "breeze" % "0.11.2",
    "org.scalanlp" %% "breeze-natives" % "0.11.2"
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", _@_*) => MergeStrategy.discard
  case _ => MergeStrategy.first
}

