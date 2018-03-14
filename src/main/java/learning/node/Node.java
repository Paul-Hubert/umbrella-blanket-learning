package learning.node;

import learning.tensor.Tensor;
import learning.optimize.Optimizer;
import learning.operation.Operation;
import learning.Network;

public class Node {
   
   protected Network net;
   
   protected Node next, before;
   
   protected Tensor input, in, output, dO;
   
   protected int[] outputSize;
   
   protected int time = 0;
   protected boolean ready = false;
   protected String name;
   
   void forwardProp() {
      output = in;
   }
   
   void backwardProp() {
      input = dO;
   }
   
   Tensor forwards(Operation op) {
      in = before.forwards(op);
      if(op.calculate) forwardProp();
      op.operate(this,output);
      return output;
   }
   
   void backwards(Tensor t, Operation op) {
      dO = t;
      if(op.calculate) backwardProp();
      op.operate(this,input);
      before.backwards(input, op);
   }
   
   protected void attachNext(Node n) {
      next = n;
   }
   
   protected void attachBefore(Node n) {
      before = n;
      setOutputSize();
      prepare();
   }
   
   public void optimize(Optimizer opt) {
      //overriden for nodes with parameters;
   }
   
   public int[] getOutputSize() {
      return outputSize;
   }
   
   protected void setOutputSize() {
      outputSize = before.getOutputSize(); // is automatically called if attachBefore is not overriden
   }
   
   protected void prepare() {
     //overriden for Tensor creation and kernel compiling
   }
   
   public Tensor forwardOp(Operation opt) {
      return this.forwards(opt);
   }
   
   public void backwardOp(Tensor t, Operation opt) {
      this.backwards(t,opt);
   }
   
   public Tensor getOutput() {
      return output;
   }
   
   public Tensor getGradient() {
      return input;
   }
   
   public void setNetwork() {
      net = Network.getCurrent();
      net.add(this);
   }
   
   public Node chain(Node n) {
      this.attachNext(n);
      n.attachBefore(this);
      setNetwork();
      return n;
   }
   
   public Node name(String s) {
      name = s;
      return this;
   }
   
}