import scala.io.Source
import BIDMat.MatFunctions.{row, col, saveAs}
import BIDMat.FMat
import scala.collection.mutable.ListBuffer
import Classifiers._
import scala.math

package Main {
  object harness {

    def main(args: Array[String]) {
      val X_train: ListBuffer[FMat] = new ListBuffer[FMat]()
      val Y_train: ListBuffer[FMat] = new ListBuffer[FMat]()

      var i: Int = 0
      for( line <- Source.fromFile("train_cleaned.csv").getLines() ) {
        if (i > 0) {
          val r = line.split(",").map(x => x.toFloat)
          val (example, label) = r.splitAt(r.length-1)
          X_train += FMat(row(example.map(x => math.log(math.abs(x.toFloat) + 0.000001))))
          Y_train += FMat(col(label))
        }
        i += 1
        if (1%10000 == 0) { println(i + " rows consumed") }
      }
      println("" + X_train.size + " rows of data")
      val classifier = new LRClassifier(X_train, Y_train, 0.00001d, 0.0001)
      println("The labels: " + Y_train)
      classifier.train()
      println("Weights: " + classifier.weights)

      saveAs("weights.mat", classifier.weights) 

    }

  }
}
