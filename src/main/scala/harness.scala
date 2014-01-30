import scala.io.Source
import BIDMat.MatFunctions.{row, col}
import BIDMat.DMat
import scala.collection.mutable.ListBuffer
import Classifiers._

package Main {
  object harness {

    def main(args: Array[String]) {
      val X_train: ListBuffer[DMat] = new ListBuffer[DMat]()
      val Y_train: ListBuffer[DMat] = new ListBuffer[DMat]()

      var i: Int = 0
      for( line <- Source.fromFile("train_cleaned.csv").getLines() ) {
        if (i > 0) {
          val r = line.split(",").map(x => x.toFloat)
          val (example, label) = r.splitAt(r.length-1)
          X_train += row(example)
          Y_train += col(label)
        }
        i += 1
        if (1%10000 == 0) { println(i + " rows consumed") }
      }
      println("" + X_train.size + " rows of data")
      val classifier = new LRClassifier(X_train, Y_train, 0.0000000000000000000000000000000000000001d, 0.0001)
      println("The labels: " + Y_train)
      classifier.train()
      println("Weights: " + classifier.weights)

    }

  }
}
