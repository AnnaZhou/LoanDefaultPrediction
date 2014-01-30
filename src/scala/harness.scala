import scala.io.Source
import BIDMat.FMat
import scala.collection.mutable.ListBuffer
import Classifiers._

object harness {
  def main(args: Array[String]) {
    for( line <- Source.fromPath(args(1)).getLines() ) {
      println(line.split(", "))
    }
  }
}
