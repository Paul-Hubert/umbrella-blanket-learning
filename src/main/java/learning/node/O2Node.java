package learning.node;

import learning.tensor.Tensor;
import learning.operation.Operation;

public class O2Node extends Node {
   
   protected Node next2;
   
   protected Tensor output2, dO2;
   
   protected boolean calculated = false;
   
   void forwardProp() {
      output = in;
      output2 = in;
   }
   
   void backwardProp() {
      input = dO;
   }
   
   Tensor forwards(Operation op) {
      if(!calculated) {
         in = before.forwards(op);
         if(op.calculate) forwardProp();
         calculated = true;
      }
      if(op.getLast() == next) {
         op.operate(this,output);
         return output;
      } if(op.getLast() == next2) {
         op.operate(this,output2);
         return output2;
      }
   }
   
   void backwards(Tensor t, Operation op) {
      dO = t;
      if(op.calculate) backwardProp();
      op.operate(this,input);
      before.backwards(input, op);
   }
   
   protected void attachNext(Node n) {
      if(next==null) {
         next = n;
      } else {
         next2 = n;
      }
   }
   
}