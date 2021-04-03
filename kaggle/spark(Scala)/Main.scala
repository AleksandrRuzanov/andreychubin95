import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession, functions => f}
import org.apache.spark.sql.types.{DoubleType, LongType}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


import scala.collection.mutable

object Main {

  // прописываю свой трансформер, на случай применение на другой набор данных (например, отдельно train и test)

  abstract class CustomTransformer{
    def transform(dataFrame: DataFrame): DataFrame
  }

  object CustomTransformer{
    def apply:CustomTransformer = (dataFrame: DataFrame) => {
      var df = dataFrame

      // создаю словарь для замены некоторых пробелов (по итогу анализа датасета в Jupyter Notebook)

      val fillMap: Map[String, String] = Map("gender" -> "Male",
        "enrolled_university" -> "no_enrollment",
        "education_level" -> "Graduate",
        "major_discipline" -> "STEM",
        "company_type" -> "Other",
        "last_new_job" -> "1")

      df = df.na.fill(fillMap)

      // создаю функцию для поска самых частовстречающихся размеров компаний для каждого типа компаний
      // Функция TOP как в SQL в Saprk не реализована
      // Предпологает широкую трансформацию .sort, что может быть неэффективно на больших данных

      def getTopInGroup(group: String, DataFrame: DataFrame = df): Option[String] = {
        try {
          Some(DataFrame.where(f.col("company_type").equalTo(group))
            .groupBy("company_size").count().sort(f.col("count").desc).first()(0).toString)
        }
        catch {
          case e: Exception => None
        }
      }

      // преобразовываю начальный датафрейм (синтаксис, к сожалению, получился очень громоздким)

      df = df.select(f.col("enrollee_id").cast(LongType),
        f.col("city"),
        f.col("city_development_index").cast(DoubleType),
        f.col("gender"),
        f.when(f.col("relevent_experience").equalTo("Has relevent experience"), 1)
          .otherwise(0).alias("relevent_experience"),
        f.col("enrolled_university"),
        f.col("education_level"),
        f.col("major_discipline"),
        f.when(f.col("experience").isNull &&
          f.col("relevent_experience").equalTo(1), "5")
          .when(f.col("experience").isNull &&
            f.col("relevent_experience").equalTo(0), "0")
          .when(f.col("experience").equalTo(">20"), "25")
          .when(f.col("experience").equalTo("<1"), "0.5")
          .otherwise(f.col("experience")).cast(DoubleType).alias("experience"),
        f.when(f.col("company_size").isNull &&
          f.col("company_type").equalTo("Funded Startup"),
          getTopInGroup("Funded Startup").getOrElse("50-99"))
          .when(f.col("company_size").isNull && f.col("company_type").equalTo("Pvt Ltd"),
            getTopInGroup("Pvt Ltd").getOrElse("50-99"))
          .when(f.col("company_size").isNull &&
            f.col("company_type").equalTo("Early Stage Startup"),
            getTopInGroup("Early Stage Startup").getOrElse("50-99"))
          .when(f.col("company_size").isNull && f.col("company_type").equalTo("NGO"),
            getTopInGroup("NGO").getOrElse("50-99"))
          .when(f.col("company_size").isNull && f.col("company_type").equalTo("Other"),
            getTopInGroup("Other").getOrElse("50-99"))
          .otherwise(f.col("company_size")).alias("company_size"),
        f.col("company_type"),
        f.col("last_new_job"),
        f.col("training_hours").cast(DoubleType),
        f.col("target").cast(DoubleType))

      // проверяю, остались ли ещё пустые значения

      val nulls = for {
        column <- df.columns
      } yield df.where(f.col(column).isNull).count()

      if (nulls.sum != 0) {
        val emergencyFillMap = Map("experience" -> 21, "company_size" -> "50-99")
        df = df.na.fill(emergencyFillMap)
      }

      // для размера фирмы извлекаю только последнее значение в группе, чтобы переквалифицировать
      // колонку из категориальной в колличественную

      val numExtractor = f.udf((x: String) => {
        def extractor[T](x: Array[T]): Int = x match {
          case _ if x.length == 2 => x(1).toString.toInt
          case _ => x.head.toString.toInt // просто .toInt не сработает
        }
        val re = raw"\d+".r
        extractor(re.findAllIn(x).toArray)
      })

      df.withColumn("company_size", numExtractor(f.col("company_size")))
    }
  }
  // находим лучшее количество деревьев для "случайного лесе" из списка с количеством деревьев

  final class BestParamsSearch(model: RandomForestClassifier, trainData: DataFrame,
                         listOfNumTrees: List[Int], setMetric: String = "areaUnderROC"){

    // копирую вводные, чтобы spark не надо было передавать весь объект

    val rf: RandomForestClassifier = this.model
    val train: DataFrame = this.trainData
    val listN: List[Int] = this.listOfNumTrees
    val metric: String = this.setMetric

    def getBestN: Int = {
      var list : mutable.MutableList[Double] = mutable.MutableList()

      for (n <- listN) {
        val rfModel = rf.setNumTrees(n).fit(train)
        val prediction = rfModel.transform(train)

        val evaluation = new BinaryClassificationEvaluator()
          .setLabelCol("target")
          .setRawPredictionCol("rawPrediction")
          .setMetricName(metric)

        list = list :+ evaluation.evaluate(prediction)
      }
      listN(list.indexOf(list.max))
    }
  }

  def main (args: Array[String]){
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val path = "<путь до папки>/Kaggle/HR Analytics Job Change of Data Scientists/archive/aug_train.csv"

    val spark = SparkSession.builder
      .appName("Spark Scala App")
      .config("spark.master", "local")
      .getOrCreate()

    var df_train: DataFrame = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", ",")
      .load(path)

    // создаю список категориальных колонок для трансформеров

    val categoricalColumns: Array[String] = Array("city", "gender", "enrolled_university",
      "education_level", "major_discipline", "company_type", "last_new_job")

    // создаю список количественных колонок

    val numericalColumns = df_train.columns
      .filterNot(categoricalColumns.contains(_))
      .filterNot(_.contains("target"))
      .filterNot(_.contains("enrollee_id"))

    // объявляю трансформеры

    val transformer = CustomTransformer.apply

    val indexer = new StringIndexer()
      .setInputCols(categoricalColumns)
      .setOutputCols(categoricalColumns.map(x => x + "_ind"))

    val encoder = new OneHotEncoder()
      .setInputCols(indexer.getOutputCols)
      .setOutputCols(categoricalColumns.map(x => x + "_ohe"))
      .setDropLast(false)

    val assembler = new VectorAssembler()
      .setInputCols(numericalColumns ++ encoder.getOutputCols)
      .setOutputCol("features")

    df_train = transformer.transform(df_train)

    // создаю пайплайн с встроенными трансформерами

    val pipeline = new Pipeline()
      .setStages(Array(indexer, encoder, assembler))
      .fit(df_train)

    df_train = pipeline.transform(df_train).select("enrollee_id", "features", "target")

    // со sparse-вектором модель в данном случае везде проставлляет 0, вне зависимости от данных
    // поэтому я преобразовываю features в dense-вектор

    val asDense = f.udf((v: Vector) => v.toDense)

    df_train = df_train.withColumn("features", asDense(f.col("features")))

    val Array(train, test) = df_train.randomSplit(Array(0.7, 0.3))
    
    train.cache()
    test.cache()

    // применяю модель

    val amountOfTrees: List[Int] = List(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

    val rf = new RandomForestClassifier()
      .setLabelCol("target")
      .setFeaturesCol("features")

    val search = new BestParamsSearch(rf, train, amountOfTrees)

    val rfModel = rf.setNumTrees(search.getBestN).fit(train)
    val prediction = rfModel.transform(test)

    val evaluation = new BinaryClassificationEvaluator()
      .setLabelCol("target")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    println(evaluation.evaluate(prediction)) // 0.7879818069324455

    spark.stop()
  }

}
