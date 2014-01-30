import scala.io.Source
<<<<<<< HEAD
import BIDMat.FMat
import BIDMat.MatFunctions.{row, col}
=======
import BIDMat.DMat
>>>>>>> SGD seems to diverge
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
<<<<<<< HEAD
          val r = line.split(",").map(x => x.toFloat)
          val (example, label) = r.splitAt(r.length-1)
          X_train += row(example)
          Y_train += col(label)
=======
          val splits: ListBuffer[Double] = new ListBuffer()
          for (f <- line.split(",").map(x => x.toDouble)) {
            splits.append(f)
          }
          val xarr: Array[Double] = new Array[Double](778)
          val yarr: Array[Double] = new Array[Double](1)

          splits.slice(1, splits.size-1).copyToArray(xarr)
          splits.slice(splits.size-1, splits.size).copyToArray(yarr)
          val xmat: DMat = new DMat(1, splits.size-2, xarr)
          val ymat: DMat = new DMat(1, 1, yarr)
          X_train.append(xmat)
          Y_train.append(ymat)
>>>>>>> SGD seems to diverge
        }
        i += 1
        if (1%10000 == 0) { println(i + " rows consumed") }
      }
      println("" + X_train.size + " rows of data")
      
<<<<<<< HEAD
      val classifier = new LRClassifier(X_train, Y_train, 0.00000001, 0.0001)
=======
      val classifier = new LRClassifier(X_train, Y_train, 0.0000000000000000000000000000000000000001d, 0.0001)
      println("The labels: " + Y_train)
>>>>>>> SGD seems to diverge
      classifier.train()
      println("Weights: " + classifier.weights)

    }

  }
}
