package spark.mlhelp
package ml

import agonda.hpopt.BayesianHyperparameterOptimization.BayesianHyperparameterOptimization
import agonda.hpopt.hyperparam.{DoubleHyperparameterType, HyperparameterEvalPoint, HyperparameterSpace, IntHyperparameterType, LinearScale, LogScale}

import org.apache.spark.ml.evaluation.{
  BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
}

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import spark.mlhelp.ml.utils.Types.{BoosterType, ModelType, EvaluatorType}

import scala.annotation.tailrec


trait TunerTrait extends LGBMParamTrait {
  var model: BoosterType

  private def evaluator: EvaluatorType = this.objective match {
    case "binary" =>
      val eval: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
        .setLabelCol(this.labelCol)
        .setRawPredictionCol(this.rawPredictionCol)
      this.evalMetric = eval.getMetricName
      eval

    case "multiclass" =>
      val eval: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol(this.labelCol)
      this.evalMetric = eval.getMetricName
      eval

    case "regression" =>
      val eval: RegressionEvaluator = new RegressionEvaluator()
        .setLabelCol(this.labelCol)
      this.evalMetric = eval.getMetricName
      eval
  }

  var bestCVMetricValue: Double = 0.0

  private val definedSpace: HyperparameterSpace = {
    HyperparameterSpace(Array(
      IntHyperparameterType("numLeaves", LinearScale(2, 2^8)),
      DoubleHyperparameterType("featureFraction", LinearScale(0.95, 1.0)),
      DoubleHyperparameterType("lambdaL1", LogScale(1e-8, 10.0)),
      DoubleHyperparameterType("lambdaL2", LogScale(1e-8, 10.0)),
      DoubleHyperparameterType("baggingFraction", LinearScale(0.95, 1.0)),
      IntHyperparameterType("baggingFreq", LinearScale(1, 7)),
      DoubleHyperparameterType("minSumHessianInLeaf", LogScale(0.0005, 0.01))
    ))
  }

  private val bho = new BayesianHyperparameterOptimization(
    definedSpace,
    seed = Some(this.seed),
    maximize = this.isHigherBetter
  )

  def crossValid(hp: HyperparameterEvalPoint, cv: Int, data: DataFrame): Double = {
    val newClassifier: BoosterType = hp.copyPipelineStage(this.model) match {
      case lgbm: BoosterType => lgbm
    }

    newClassifier
      .setLabelCol(this.labelCol)
      .setFeaturesCol(this.featuresCol)

    val aucSeq = (0 until cv).map(split => {
      val train = data.where(col("split") =!= split)
      val eval = data.where(col("split") === split)
      evaluator.evaluate(newClassifier.fit(train).transform(eval))
    })

    aucSeq.sum / cv
  }

  def tune(iterations: Int, randomIterations: Int = 0, cv: Int, data: DataFrame,
           preExistentHP: Array[HyperparameterEvalPoint] = Array.empty[HyperparameterEvalPoint],
           preExistentMetrics: Array[Double] = Array.empty[Double]): Unit = {

    @tailrec
    def iterationStep(step: Int, previousHP: Array[HyperparameterEvalPoint],
                      previousMetrics: Array[Double]): (Array[HyperparameterEvalPoint], Array[Double]) = {

      val nextHP = if (step < randomIterations) bho.getNextRandom else bho.getNext(previousHP, previousMetrics)
      val nextMetric = crossValid(hp = nextHP, cv = cv, data = data)
      val allHP = previousHP :+ nextHP
      val allMetrics: Array[Double] = previousMetrics :+ nextMetric

      if (step < iterations - 1) iterationStep(step + 1, allHP, allMetrics)
      else (allHP, allMetrics)
    }
    val (allHP, allMetrics) = iterationStep(0, preExistentHP, preExistentMetrics)
    val paramMap = allHP.map(_.hyperparameters.map(_.value).toArray)
    val resultParams: (Array[Any], Double) = (paramMap zip allMetrics).maxBy(x => x._2)
    val result: Array[Any] = resultParams._1
    bestCVMetricValue = resultParams._2

    this.numLeaves = result(0).toString.toInt
    this.featureFraction = result(1).toString.toDouble
    this.lambdaL1 = result(2).toString.toDouble
    this.lambdaL2 = result(3).toString.toDouble
    this.baggingFraction = result(4).toString.toDouble
    this.baggingFreq = result(5).toString.toInt
    this.minSumHessianInLeaf = result(6).toString.toDouble
  }

  def showEndParams: String = {
      s"""
         |BEST MODEL PARAMS:
         |labelCol = ${this.model.getLabelCol}
         |featureCol = ${this.model.getFeaturesCol}
         |catCols = ${this.model.getCategoricalSlotIndexes.mkString(", ")}
         |numIterations = ${this.model.getNumIterations}
         |numLeaves = ${this.model.getNumLeaves}
         |lambda1 = ${this.model.getLambdaL1}
         |lambda2 = ${this.model.getLambdaL2}
         |featureFraction = ${this.model.getFeatureFraction}
         |baggingFraction = ${this.model.getBaggingFraction}
         |baggingFreq = ${this.model.getBaggingFreq}
         |baggingSeed = ${this.model.getBaggingSeed}
         |minSumHessianInLeaf = ${this.model.getMinSumHessianInLeaf}
         |
         |Best cross-validation score = ${this.bestCVMetricValue}
         |Validation metric: ${this.evalMetric}
         |""".stripMargin
  }

  def fit(dataFrame: DataFrame): ModelType

  def fit(dataFrame: DataFrame, tuneIterations: Int,
          randomIterations: Int, cv: Int): ModelType
}
