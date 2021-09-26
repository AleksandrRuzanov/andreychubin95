package spark.mlhelp

import scala.reflect.ClassTag
import scala.util.Random
import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType


final class RandomPartition[A: ClassTag](val index: Int, numValues: Int, random: => A) extends Partition {
  def values: Iterator[A] = Iterator.fill(numValues)(random)
}

final class RandomRDD[A: ClassTag](@transient private val sc: SparkContext,
                             numSlices: Int,
                             numValues: Int,
                             random: => A) extends RDD[A](sc, deps = Seq.empty) {

  private val valuesPerSlice = numValues / numSlices
  private val slicesWithExtraItem = numValues % numSlices

  override def compute(split: Partition, context: TaskContext): Iterator[A] =
    split.asInstanceOf[RandomPartition[A]].values

  override protected def getPartitions: Array[Partition] =
    ((0 until slicesWithExtraItem).view.map(new RandomPartition[A](_, valuesPerSlice + 1, random)) ++
      (slicesWithExtraItem until numSlices).view.map(new RandomPartition[A](_, valuesPerSlice, random))).toArray
}

class RandomDataFrameCreator(@transient private val sparkSession: SparkSession) extends java.io.Serializable {
  @transient private val spark: SparkSession = this.sparkSession
  @transient private val sc: SparkContext = spark.sparkContext

  type RandomRow = (Double, Double, Double, Double, Int, Int, Int, Double)

  private def randomDouble: Double = Random.nextDouble() * Random.nextInt(100).toDouble
  private def randomInt: Int = Random.nextInt(10)
  private def randomTiny: Int = Random.nextInt(2)

  def structure(targetType: String): RandomRow =
    targetType match {
      case "binary" =>
        (
          randomDouble, randomDouble, randomDouble, randomDouble,
          randomTiny, randomTiny, randomTiny, randomTiny.toDouble
        )
      case "multiclass" =>
        (
          randomDouble, randomDouble, randomDouble, randomDouble,
          randomTiny, randomTiny, randomTiny, randomInt.toDouble
        )
      case "regression" =>
        (
          randomDouble, randomDouble, randomDouble, randomDouble,
          randomTiny, randomTiny, randomTiny, randomDouble
        )
  }

  private def createRandomRDD(numRows: Int, numPartitions: Int, targetType: String): RandomRDD[RandomRow] = {
    new RandomRDD(sc, numPartitions, numRows, structure(targetType))
  }

  def build(numRows: Int, numPartitions: Int, targetType: String): DataFrame = {
    val rdd: RandomRDD[RandomRow] = createRandomRDD(numRows, numPartitions, targetType)
    var df: DataFrame = spark.createDataFrame(rdd)
    if (targetType != "regression")
      df = df.withColumn("_8", col("_8").cast(IntegerType))
    df.select(
      col("_1").alias("double_0"),
      col("_2").alias("double_1"),
      col("_3").alias("double_2"),
      col("_4").alias("double_3"),
      col("_5").alias("ohe_cat_0"),
      col("_6").alias("ohe_cat_1"),
      col("_7").alias("ohe_cat_2"),
      col("_8").alias("target"))
  }
}

object SparkStart{
  def apply(appName: String, master: String = "local"): SparkSession = {
    val spark: SparkSession = SparkSession.builder
      .appName(appName)
      .config("spark.master", master)
      .getOrCreate()

    spark
  }
}

private object DataCreator extends App {
  val spark: SparkSession = SparkStart("CreatorTest")
  val creator: RandomDataFrameCreator = new RandomDataFrameCreator(spark)
  val randomDF: DataFrame = creator.build(300, 10, "binary")
  randomDF.show()
  spark.stop()
}
