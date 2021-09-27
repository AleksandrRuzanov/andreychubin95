package spark.mlhelp
package ml

import org.apache.log4j.Logger
import com.microsoft.ml.spark.lightgbm.{LightGBMClassifier, LightGBMRegressor}
import org.apache.spark.sql.DataFrame
import spark.mlhelp.ml.utils.Types.{BoosterType, ModelType}

import scala.annotation.meta.setter


trait LGBMTuner extends TunerTrait with java.io.Serializable {
  // java.io.Serializable with
  override var model: BoosterType =  new LightGBMClassifier().setVerbosity(-1)

  @setter
  def setLabelCol(colName: String): this.type = {
    this.labelCol = colName
    this.model.setLabelCol(colName)
    this
  }

  @setter
  def setFeaturesCol(colName: String): this.type = {
    this.featuresCol = colName
    this.model.setFeaturesCol(colName)
    this
  }

  @setter
  def setNumIterations(num: Int): this.type = {
    this.numIterations = num
    this.model.setNumIterations(num)
    this
  }

  @setter
  def setCategoricalSlotIndexes(indexes: Array[Int]): this.type = {
    this.catColumnsIndexes = indexes
    this.model.setCategoricalSlotIndexes(indexes)
    this
  }

  override def fit(dataFrame: DataFrame): ModelType = {
    Logger.getRootLogger.info("Fitting with no HyperOptSearch was chosen")
    this.model.fit(dataFrame)
  }

  override def fit(dataFrame: DataFrame, tuneIterations: Int,
                   randomIterations: Int = 10, cv: Int = 5): ModelType = {
    Logger.getRootLogger.info("Fitting with HyperOptSearch was chosen")

    tune(
      iterations = tuneIterations, randomIterations = randomIterations, cv = cv, data = dataFrame
    )

    this.model = new LightGBMClassifier()
      .setVerbosity(-1)
      .setLabelCol(this.labelCol)
      .setFeaturesCol(this.featuresCol)
      .setNumIterations(this.numIterations)
      .setNumLeaves(this.numLeaves)
      .setLambdaL1(this.lambdaL1)
      .setLambdaL2(this.lambdaL2)
      .setFeatureFraction(this.featureFraction)
      .setBaggingFraction(this.baggingFraction)
      .setBaggingFreq(this.baggingFreq)
      .setMinSumHessianInLeaf(this.minSumHessianInLeaf)
      .setBaggingSeed(this.seed.toInt)
      .setUseBarrierExecutionMode(true)

    if (!this.catColumnsIndexes.isEmpty) model.setCategoricalSlotIndexes(this.catColumnsIndexes)

    Logger.getRootLogger.info(showEndParams)

    Logger.getRootLogger.info("Started fitting to best model")

    model.fit(dataFrame)
  }
}

class LGBMBinaryTuner extends LGBMTuner {
  override val objective: String = "binary"
}

object LGBMBinaryTuner {
  def apply(): LGBMBinaryTuner = new LGBMBinaryTuner
}

class LGBMMulticlassTuner extends LGBMTuner {
  override val objective: String = "multiclass"
}

object LGBMMulticlassTuner {
  def apply(): LGBMMulticlassTuner = new LGBMMulticlassTuner
}

class LGBMRegressionTuner extends LGBMTuner {
  override val objective: String = "regression"
  override var model: BoosterType =  new LightGBMRegressor().setVerbosity(-1)

  override def fit(dataFrame: DataFrame): ModelType = {
    Logger.getRootLogger.info("Fitting with no HyperOptSearch was chosen")
    this.model.fit(dataFrame)
  }

  override def fit(dataFrame: DataFrame, tuneIterations: Int,
                   randomIterations: Int = 10, cv: Int = 5): ModelType = {
    Logger.getRootLogger.info("Fitting with HyperOptSearch was chosen")

    tune(
      iterations = tuneIterations, randomIterations = randomIterations, cv = cv, data = dataFrame
    )

    this.model = new LightGBMRegressor()
      .setVerbosity(-1)
      .setLabelCol(this.labelCol)
      .setFeaturesCol(this.featuresCol)
      .setNumIterations(this.numIterations)
      .setNumLeaves(this.numLeaves)
      .setLambdaL1(this.lambdaL1)
      .setLambdaL2(this.lambdaL2)
      .setFeatureFraction(this.featureFraction)
      .setBaggingFraction(this.baggingFraction)
      .setBaggingFreq(this.baggingFreq)
      .setMinSumHessianInLeaf(this.minSumHessianInLeaf)
      .setBaggingSeed(this.seed.toInt)
      .setUseBarrierExecutionMode(true)

    if (!this.catColumnsIndexes.isEmpty) model.setCategoricalSlotIndexes(this.catColumnsIndexes)

    Logger.getRootLogger.info(showEndParams)

    Logger.getRootLogger.info("Started fitting to best model")

    model.fit(dataFrame)
  }
}

object LGBMRegressionTuner {
  def apply(): LGBMRegressionTuner = new LGBMRegressionTuner
}
