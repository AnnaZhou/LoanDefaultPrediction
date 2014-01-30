import BIDMat.DMat
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import scala.collection.mutable.ListBuffer
import scala.math

package Classifiers {

  object lineTest {
    def main(args: Array[String]) {
      val x:DMat = (1 \ 1 \ 1) on (2 \ 2 \ 2) on (3 \ 3 \ 3)
      val y:DMat = 1 on 2 on 3
      val c = new LRClassifier(new ListBuffer() += x, new ListBuffer() += y, 0.001, 0.0001)
      c.train()
      println("examples: ")
      println(x)
      println("labels: ")
      println(y)
      println("Learned Weights:")
      println(c.weights)
    }
  }


  class LRClassifier(examples: ListBuffer[DMat], labels: ListBuffer[DMat], a: Double, t: Double) {
    val numFeatures = examples(0).ncols
    val trainingSet = examples.zip(labels)
    var weights: DMat = zeros(numFeatures, 1) //single column of weights, with numRows = numFeatures
    var alpha: Double = a
    
    def gradients(W:DMat, X:DMat, Y:DMat):DMat = {
      //println("x*w: " + X*W)
      //println("-y: " + (X*W-Y))
      //println("t: " + (X*W-Y).t)
      //println("*2*x: " + ((X*W-Y).t * (2*X)))
      //println(".t: " + ((X*W-Y).t * (2*X)).t)

      val r: DMat = (((X*W) - Y).t * (2*X)).t
      return r

    }

    def error(gs:DMat): Double = sum(abs(gs), 1)(0,0)

    def train() {
      var it: Int = 1
        var err = 0.0d
        for ( (x,y) <- trainingSet ) {
          println("x: " + x + ", y: " + y)
          //println("row: " + x)
          val gs = gradients(weights, x, y)
          println("Gradients: " + gs)
          err += error(gs)
          println("Error: " + err)
          weights -= gs * alpha
        }
        if (err > t) {
          println("Error: " + err)
          println("iteration number " + it)
          train()
        }
    }

    def predict(x: DMat): Double = (x*weights)(0,0)
  }

}
