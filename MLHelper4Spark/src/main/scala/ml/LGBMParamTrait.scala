package spark.mlhelp
package ml

trait LGBMParamTrait {
  val objective: String
  var evalMetric: String = "default"
  var labelCol: String = "label"
  var featuresCol: String = "features"
  var numIterations: Int = 50
  var rawPredictionCol: String = "rawPrediction"
  var catColumnsIndexes: Array[Int] = Array()
  // add addition of categorical columns by column names
  var seed: Long = 24112020

  var numLeaves: Int = 31
  var lambdaL1: Double = 0.0
  var lambdaL2: Double = 0.0
  var featureFraction: Double = 1.0
  var baggingFraction: Double = 1.0
  var baggingFreq: Int = 0
  var minSumHessianInLeaf: Double = 0.001

  val isHigherBetter: Boolean = true
}
