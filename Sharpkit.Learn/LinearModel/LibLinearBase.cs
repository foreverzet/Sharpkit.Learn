// -----------------------------------------------------------------------
// <copyright file="LibLinearBase.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Liblinear;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Sharpkit.Learn.Preprocessing;

    /// <summary>
    /// Base for classes binding liblinear (dense and sparse versions).
    /// </summary>
    internal class LibLinearBase<TLabel> where TLabel : IEquatable<TLabel>
    {
        private readonly SolverType solverType;
        public double tol { get; private set; }
        public double C { get; private set; }
        public double interceptScaling { get; private set; }
        private readonly ClassWeight<TLabel> classWeight;
        private readonly int verbose;
        private Random random;
        private LabelEncoder<TLabel> _enc;

        private Model model;
        private readonly LinearClassifier<TLabel> classifier;
        private bool dual;

        public LibLinearBase(
            LinearClassifier<TLabel> classifier,
            Norm norm = Norm.L2,
            Loss loss = Loss.L2,
            bool dual = true,
            double tol = 1e-4,
            double C = 1.0,
            Multiclass multiclass = Multiclass.Ovr,
            double interceptScaling = 1,
            ClassWeight<TLabel> classWeight = null,
            int verbose = 0,
            Random random = null)
        {
            this.solverType = GetSolverType(norm, loss, dual, multiclass);

            this.classifier = classifier;
            this.tol = tol;
            this.C = C;
            this.interceptScaling = interceptScaling;
            this.classWeight = classWeight ?? ClassWeight<TLabel>.Uniform;
            this.verbose = verbose;
            this.random = random;
            this.dual = dual;
        }

        private SolverType GetSolverType(Norm norm, Loss loss, bool dual, Multiclass multiclass)
        {
            if (multiclass == Multiclass.CrammerSinger)
                return SolverType.getById(SolverType.MCSVM_CS);
            
            if (multiclass != Multiclass.Ovr)
                throw new ArgumentException("Invalid multiclass value");

            if (norm == Norm.L2 && loss == Loss.LogisticRegression && !dual)
                return SolverType.getById(SolverType.L2R_LR);
            
            if (norm == Norm.L2 && loss == Loss.L2 && dual)
                return SolverType.getById(SolverType.L2R_L2LOSS_SVC_DUAL);

            if (norm == Norm.L2 && loss == Loss.L2 && !dual)
                return SolverType.getById(SolverType.L2R_L2LOSS_SVC);

            if (norm == Norm.L2 && loss == Loss.L1 && dual)
                return SolverType.getById(SolverType.L2R_L1LOSS_SVC_DUAL);

            if (norm == Norm.L1 && loss == Loss.L2 && !dual)
                return SolverType.getById(SolverType.L1R_L2LOSS_SVC);

            if (norm == Norm.L1 && loss == Loss.LogisticRegression && !dual)
                return SolverType.getById(SolverType.L1R_LR);

            if (norm == Norm.L2 && loss == Loss.LogisticRegression && dual)
                return SolverType.getById(SolverType.L2R_LR_DUAL);

            throw new ArgumentException("Given combination of penalty, loss, dual params is not supported");
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="x">shape = [n_samples, n_features]
        ///    Training vector, where n_samples in the number of samples and
        ///    n_features is the number of features.</param>
        /// <param name="y">shape = [n_samples]
        ///    Target vector relative to X</param>
        public void Fit(Matrix x, TLabel[] y)
        {
            Linear.random = this.random;
            
            this._enc = new LabelEncoder<TLabel>();
            int[] y_ = this._enc.FitTransform(y);
            if (this.Classes.Length < 2)
                throw new ArgumentException("The number of classes has to be greater than one.");
            
            Vector class_weight = this.classWeight.ComputeWeights(this.Classes, y_);
            
            if (x.RowCount != y.Length)
            {
                throw new ArgumentException(
                    string.Format("X and y have incompatible shapes.\n X has {0} samples, but y has {1}.",
                                  x.RowCount,
                                  y.Length));

            }

            if (this.verbose > 0)
            {
                Console.WriteLine("[LibLinear]");
            }

            Problem problem = new Problem();
            problem.bias = this.Bias;
            var samples = new List<Feature>[x.RowCount];
            for (int i=0; i<samples.Length; i++)
            {
                samples[i] = new List<Feature>();
            }
            foreach (var i in x.IndexedEnumerator())
            {
                samples[i.Item1].Add(new Feature(i.Item2 + 1, i.Item3));
            }

            if (this.Bias > 0)
            {
                for (int i = 0; i < x.RowCount; i++)
                    samples[i].Add(new Feature(x.ColumnCount + 1, this.Bias));
            }

            problem.x = samples.Select(s => s.ToArray()).ToArray();
            problem.y = y_.Select( v => (double)v).ToArray();
            problem.l = x.RowCount;
            problem.n = this.Bias > 0 ? x.ColumnCount + 1 : x.ColumnCount;
            
            Parameter prm = new Parameter(this.solverType, this.C, this.tol);
            prm.weightLabel = Enumerable.Range(0, this.Classes.Length).ToArray();
            prm.weight = class_weight.ToArray();

            this.model = Linear.train(problem, prm);
            
            //
            int nr_class = model.getNrClass();
            int nr_feature = model.getNrFeature();
            if (Bias > 0)
            {
                nr_feature = nr_feature + 1;
            }

            Matrix r;
            if (nr_class == 2)
            {
                r = DenseMatrix.OfColumnMajor(1, nr_feature, model.getFeatureWeights());
            }
            else
            {
                r = DenseMatrix.OfColumnMajor(nr_class, nr_feature, model.getFeatureWeights());
            }

            if (nr_class > 2)
            {
                var rClone = r.Clone();
                var labels = model.getLabels();
                for (int i = 0; i < labels.Length; i++)
                {
                    rClone.SetRow(labels[i], r.Row(i));
                }
                r = (Matrix)rClone;
            }
            else
            {
                r.MapInplace(v => v * -1);
            }

            if (this.classifier.FitIntercept)
            {
                this.classifier.CoefMatrix = (Matrix)r.SubMatrix(0, r.RowCount, 0, r.ColumnCount - 1).Transpose();
                this.classifier.InterceptVector = (Vector)(r.Column(r.ColumnCount - 1) * this.interceptScaling);
            }
            else
            {
                this.classifier.CoefMatrix = (Matrix)r.Transpose();
                this.classifier.InterceptVector = new DenseVector(r.ColumnCount);
            }
        }
    
    public TLabel[] Classes
    {
        get { return this._enc.Classes; }
    }

    public bool Dual { get { return this.dual; } }
    

    private double Bias
    {
        get
        {
            if (this.classifier.FitIntercept)
                return this.interceptScaling;
            else
            {
                return -1.0;
            }
        }
    }
    }
}
