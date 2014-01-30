import scala.io.Source
import BIDMat.FMat
import scala.collection.mutable.ListBuffer
import Classifiers._

package Main {
  object harness {
    def main(args: Array[String]) {
      for( line <- Source.fromFile(args(1)).getLines() ) {
        println(line.split(", "))
      }
    }
  }
}
