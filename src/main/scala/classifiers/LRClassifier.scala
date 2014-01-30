import BIDMat.FMat
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import scala.collection.mutable.ListBuffer
import scala.math

import scala.io.Source

package Classifiers {

  object runner {
    def main(args: Array[String]) {
      val c = new LRClassifier(0.001, 0.0001)
      c.chunks()
      c.train()

      println(c.weights)
    }

  }

  class LRClassifier(a: Float, t: Float) {
    var X_train: ListBuffer[FMat]
    var Y_train: FMat = null
    val numFeatures = X_train(0).ncols
    var weights: FMat = zeros(numFeatures, 1) //single column of weights, with numRows = numFeatures
    var alpha: Float = a

    def gradients(W:FMat, X:FMat, Y:FMat): FMat = (((X*W) - Y).t * (2*X)).t

    def error(gs:FMat): Float = sum(abs(gs), 1)(0,0)

    def train() {
      var err = 0.0f
      for (X <- X_train ) {
        val gs = gradients(weights, X, Y_train)
        err += error(gs)
        weights -= gs * alpha
      }
      if (err > t) {
        train()
      }
    }

    def predict(x: FMat): Float = (x*weights)(0,0)

    def chunks() {
      var X_train: ListBuffer[FMat] = new ListBuffer[FMat]()

      val Y: ListBuffer[Float] = new ListBuffer[Float]()
      for (line <- Source.fromFile("train.csv").getLines()) {
        val buf: ListBuffer[Float] = new ListBuffer[Float]()
        val splits: Array[String] = line.split(",")
        val dubs: Array[Float] = splits.map(s => s.toFloat())

        val mat: FMat = new FMat(1, 800, dubs.slice(1, 802))
        X_train.append(mat)
        Y.append(dubs.get(801))


      }
      var Y_train: FMat = new FMat(Y.size(), 1, Y.asArray())
      this.X_train = X_train
      this.Y_train = Y_train
    }

  }

}
