import scala.io.Source
import BIDMat.FMat
import scala.collection.mutable.ListBuffer
import Classifiers._

package Main {
  object harness {

    def main(args: Array[String]) {
      val X_train: ListBuffer[FMat] = new ListBuffer[FMat]()
      val Y_train: ListBuffer[FMat] = new ListBuffer[FMat]()

      var i: Int = 0
      for( line <- Source.fromFile("train_cleaned.csv").getLines() ) {
        if (i > 0) {
          val splits: ListBuffer[Float] = new ListBuffer()
          for (f <- line.split(",").map(x => x.toFloat)) {
            splits.append(f)
          }
          val xarr: Array[Float] = new Array[Float](778)
          val yarr: Array[Float] = new Array[Float](1)

          splits.slice(1, 779).copyToArray(xarr)
          splits.slice(779, 780).copyToArray(yarr)
          val xmat: FMat = new FMat(1, 778, xarr)
          val ymat: FMat = new FMat(1, 1, yarr)
          X_train.append(xmat)
          Y_train.append(ymat)
        }
        i += 1
      }
      println("" + X_train.size + " rows of data")
      
      val classifier = new LRClassifier(X_train, Y_train, 0.001, 0.0001)
      classifier.train()
      println("Weights: " + classifier.weights)

    }

  }
}
