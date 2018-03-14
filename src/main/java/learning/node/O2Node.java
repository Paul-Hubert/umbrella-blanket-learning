package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class O2Node extends Node {
   
   protected Node next2;
   
   protected Tensor output2, dO2;
   
   void forwardProp() {
      output = in;
   }
   
   void backwardProp() {
      input = dO;
   }
   
   Tensor forwards(Operation op) {
      return output;
   }
   
   void backwards(Tensor t, Operation op) {
      
   }
   
   protected void attachNext(Node n) {
      if(next==null) {
         next = n;
      } else {
         next2 = n;
      }
   }
   
}