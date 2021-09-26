package spark.mlhelp

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import spark.mlhelp.ml.{LGBMBinaryTuner, LGBMMulticlassTuner}
import spark.mlhelp.ml.utils.Types.ModelType


object MLHelper extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com").setLevel(Level.ERROR)

  val cvCount: Int = 3
  val objective: String = "multiclass"
  val label: String = "target"

  val spark: SparkSession = SparkStart("MLHelper")
  val dataCreator: RandomDataFrameCreator = new RandomDataFrameCreator(spark)
  var randomDF: DataFrame = dataCreator.build(300, 1, objective)
  var randomDFTest: DataFrame = dataCreator.build(150, 1, objective)
  val columns: Array[String] = randomDF.columns.filter(_ != label)
  val va: VectorAssembler = new VectorAssembler().setInputCols(columns).setOutputCol("features")
  randomDF = va.transform(randomDF)
  randomDFTest = va.transform(randomDFTest)
    .select(col("features"), col(label))

  randomDF = randomDF.withColumn("id", monotonically_increasing_id())
  val window = Window.orderBy(col("id"))
  randomDF = randomDF.withColumn("id", row_number().over(window))
    .withColumn("split", col("id") mod cvCount)
    .select(col("features"), col(label), col("split"))

  randomDF.show(1)
  randomDF.cache()

  val helper: LGBMMulticlassTuner = LGBMMulticlassTuner()
    .setFeaturesCol("features")
    .setLabelCol(label)
    .setNumIterations(10)
    .setCategoricalSlotIndexes(Array(4,5,6))

  val helperModel: ModelType = helper.fit(dataFrame = randomDF, tuneIterations = 10)
  val predDF: DataFrame = helperModel.transform(randomDFTest)

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol(helper.labelCol)
    .setRawPredictionCol(helper.rawPredictionCol)

  println(s"ROC AUC is ${evaluator.evaluate(predDF)}")

  spark.stop()
}
