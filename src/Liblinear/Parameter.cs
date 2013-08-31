// -----------------------------------------------------------------------
// <copyright file="Parameter.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
   public sealed class Parameter {


    public double     C;


    /** stopping criteria */
    public double     eps;


    public SolverType solverType;


    public double[]   weight      = null;


    public int[]      weightLabel = null;


    public double     p;


    public Parameter(SolverType solver, double C, double eps ):this(solver, C, eps, 0.1) {
        
    }


    public Parameter(SolverType solverType, double C, double eps, double p ) {
        setSolverType(solverType);
        setC(C);
        setEps(eps);
        setP(p);
    }


    /**
     * <p>nr_weight, weight_label, and weight are used to change the penalty
     * for some classes (If the weight for a class is not changed, it is
     * set to 1). This is useful for training classifier using unbalanced
     * input data or with asymmetric misclassification cost.</p>
     *
     * <p>Each weight[i] corresponds to weight_label[i], meaning that
     * the penalty of class weight_label[i] is scaled by a factor of weight[i].</p>
     *
     * <p>If you do not want to change penalty for any of the classes,
     * just set nr_weight to 0.</p>
     */
    public void setWeights(double[] weights, int[] weightLabels) {
        if (weights == null) throw new ArgumentException("'weight' must not be null");
        if (weightLabels == null || weightLabels.Length != weights.Length)
            throw new ArgumentException("'weightLabels' must have same length as 'weight'");
        this.weightLabel = Linear.copyOf(weightLabels, weightLabels.Length);
        this.weight = Linear.copyOf(weights, weights.Length);
    }


    /**
     * @see #setWeights(double[], int[])
     */
    public double[] getWeights() {
        return Linear.copyOf(weight, weight.Length);
    }


    /**
     * @see #setWeights(double[], int[])
     */
    public int[] getWeightLabels() {
        return Linear.copyOf(weightLabel, weightLabel.Length);
    }


    /**
     * the number of weights
     * @see #setWeights(double[], int[])
     */
    public int getNumWeights() {
        if (weight == null) return 0;
        return weight.Length;
    }


    /**
     * C is the cost of constraints violation. (we usually use 1 to 1000)
     */
    public void setC(double C) {
        if (C <= 0) throw new ArgumentException("C must not be <= 0");
        this.C = C;
    }


    public double getC() {
        return C;
    }


    /**
     * eps is the stopping criterion. (we usually use 0.01).
     */
    public void setEps(double eps) {
        if (eps <= 0) throw new ArgumentException("eps must not be <= 0");
        this.eps = eps;
    }


    public double getEps() {
        return eps;
    }


    public void setSolverType(SolverType solverType) {
        if (solverType == null) throw new ArgumentException("solver type must not be null");
        this.solverType = solverType;
    }


    public SolverType getSolverType() {
        return solverType;
    }




    /**
     * set the epsilon in loss function of epsilon-SVR (default 0.1)
     */
    public void setP(double p) {
        if (p < 0) throw new ArgumentException("p must not be less than 0");
        this.p = p;
    }


    public double getP() {
        return p;
    }
}

}
