// -----------------------------------------------------------------------
// <copyright file="Model.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.IO;
using System.Runtime.Serialization;

namespace Liblinear
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
   /**
 * <p>Model stores the model obtained from the training procedure</p>
 *
 * <p>use {@link Linear#loadModel(File)} and {@link Linear#saveModel(File, Model)} to load/save it</p>
 */
public sealed class Model {


    private static readonly long serialVersionUID = -6456047576741854834L;


    public double                    bias;


    /** label of each class */
    public int[]                     label;


    public int                       nr_class;


    public int                       nr_feature;


    public SolverType                solverType;


    /** feature weight array */
    public double[]                  w;


    /**
     * @return number of classes
     */
    public int getNrClass() {
        return nr_class;
    }


    /**
     * @return number of features
     */
    public int getNrFeature() {
        return nr_feature;
    }


    public int[] getLabels() {
        return Linear.copyOf(label, nr_class);
    }


    /**
     * The nr_feature*nr_class array w gives feature weights. We use one
     * against the rest for multi-class classification, so each feature
     * index corresponds to nr_class weight values. Weights are
     * organized in the following way
     *
     * <pre>
     * +------------------+------------------+------------+
     * | nr_class weights | nr_class weights |  ...
     * | for 1st feature  | for 2nd feature  |
     * +------------------+------------------+------------+
     * </pre>
     *
     * If bias &gt;= 0, x becomes [x; bias]. The number of features is
     * increased by one, so w is a (nr_feature+1)*nr_class array. The
     * value of bias is stored in the variable bias.
     * @see #getBias()
     * @return a <b>copy of</b> the feature weight array as described
     */
    public double[] getFeatureWeights() {
        return Linear.copyOf(w, w.Length);
    }


    /**
     * @return true for logistic regression solvers
     */
    public bool isProbabilityModel() {
        return solverType.isLogisticRegressionSolver();
    }


    /**
     * @see #getFeatureWeights()
     */
    public double getBias() {
        return bias;
    }

    public override String ToString() {
        StringBuilder sb = new StringBuilder("Model");
        sb.Append(" bias=").Append(bias);
        sb.Append(" nr_class=").Append(nr_class);
        sb.Append(" nr_feature=").Append(nr_feature);
        sb.Append(" solverType=").Append(solverType);
        return sb.ToString();
    }

        public bool Equals(Model other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return other.bias.Equals(bias) && other.label.SequenceEqual(label) && other.nr_class == nr_class && other.nr_feature == nr_feature && Equals(other.solverType, solverType) && other.w.SequenceEqual(w);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != typeof (Model)) return false;
            return Equals((Model)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int result = bias.GetHashCode();
                result = (result*397) ^ (label != null ? label.GetHashCode() : 0);
                result = (result*397) ^ nr_class;
                result = (result*397) ^ nr_feature;
                result = (result*397) ^ (solverType != null ? solverType.GetHashCode() : 0);
                result = (result*397) ^ (w != null ? w.GetHashCode() : 0);
                return result;
            }
        }

    /**
     * see {@link Linear#saveModel(java.io.File, Model)}
     */
    public void save(FileInfo file) {
        Linear.saveModel(file, this);
    }


    /**
     * see {@link Linear#saveModel(Writer, Model)}
     */
    public void save(StreamWriter writer) {
        Linear.saveModel(writer, this);
    }


    /**
     * see {@link Linear#loadModel(File)}
     */
    public static Model load(FileInfo file) {
        return Linear.loadModel(file);
    }


    /**
     * see {@link Linear#loadModel(Reader)}
     */
    public static Model load(StreamReader inputReader) {
        return Linear.loadModel(inputReader);
    }
}

}
