package spark.mlhelp
package ml.utils


import com.microsoft.ml.spark.core.serialize.ConstructorWritable
import com.microsoft.ml.spark.lightgbm.{LightGBMBase, LightGBMClassificationModel, LightGBMClassifier, LightGBMRegressionModel, LightGBMRegressor}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.{PredictionModel, Predictor, linalg}

object Types {
  type BoosterType = Predictor[
    linalg.Vector, _ >: LightGBMClassifier with LightGBMRegressor <: Predictor[
    linalg.Vector, _ >: LightGBMClassifier
    with LightGBMRegressor, _ >: LightGBMClassificationModel
    with LightGBMRegressionModel]
    with LightGBMBase[_ >: LightGBMClassificationModel
    with LightGBMRegressionModel], _ >: LightGBMClassificationModel
    with LightGBMRegressionModel <: PredictionModel[linalg.Vector, _ >: LightGBMClassificationModel
    with LightGBMRegressionModel]
    with ConstructorWritable[_ >: LightGBMClassificationModel
    with LightGBMRegressionModel]]
    with LightGBMBase[_ >: LightGBMClassificationModel
    with LightGBMRegressionModel <: PredictionModel[linalg.Vector, _ >: LightGBMClassificationModel
    with LightGBMRegressionModel]
    with ConstructorWritable[_ >: LightGBMClassificationModel
    with LightGBMRegressionModel]]

  type ModelType = PredictionModel[linalg.Vector, _ >: LightGBMClassificationModel with LightGBMRegressionModel]
    with ConstructorWritable[_ >: LightGBMClassificationModel with LightGBMRegressionModel]

  type EvaluatorType = Evaluator with HasLabelCol with DefaultParamsWritable
}
