// -----------------------------------------------------------------------
// <copyright file="LibLinearBase.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
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
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Base for classes binding liblinear (dense and sparse versions).
    /// </summary>
    public class LibLinearBase<TLabel> : LinearModel where TLabel : IEquatable<TLabel>
    {
        private readonly SolverType solverType;
        private double tol { get; set; }
        /// <summary>
        /// Gets or sets inverse of regularization strength; must be a positive float.
        /// Like in support vector machines, smaller values specify stronger
        /// regularization.
        /// </summary>
        public double C { get; private set; }
        private double interceptScaling { get; set; }

        /// <summary>
        /// Gets or sets class weight estimator.
        /// </summary>
        public ClassWeightEstimator<TLabel> ClassWeightEstimator { get; private set; }

        private readonly int verbose;
        private readonly Random random;
        private LabelEncoder<TLabel> _enc;

        private Model model;
        
        /// <summary>
        /// Gets or sets a value indicating whether formulation is primal or dual. Dual formulation is only
        /// implemented for l2 penalty. Prefer dual=false when
        /// nSamples > nFeatures.
        /// </summary>
        public bool Dual { get; private set; }

        internal LibLinearBase(
            bool fitIntercept,
            Norm norm = Norm.L2,
            Loss loss = Loss.L2,
            bool dual = true,
            double tol = 1e-4,
            double c = 1.0,
            Multiclass multiclass = Multiclass.Ovr,
            double interceptScaling = 1,
            ClassWeightEstimator<TLabel> classWeightEstimator = null,
            int verbose = 0,
            Random random = null) : base(fitIntercept)
        {
            this.solverType = GetSolverType(norm, loss, dual, multiclass);

            //this.fitIntercept = fitIntercept;
            this.tol = tol;
            this.C = c;
            this.interceptScaling = interceptScaling;
            this.ClassWeightEstimator = classWeightEstimator ?? ClassWeightEstimator<TLabel>.Uniform;
            this.verbose = verbose;
            this.random = random;
            this.Dual = dual;
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
        /// <param name="sampleWeight">Sample weights.</param>
        public void Fit(Matrix<double> x, TLabel[] y, Vector<double> sampleWeight = null)
        {
            if (sampleWeight != null)
            {
                throw new ArgumentException("Sample weights are not supported by the classifier");
            }

            Linear.random = this.random;

            this._enc = new LabelEncoder<TLabel>();
            int[] y1 = this._enc.FitTransform(y);
            if (this.Classes.Length < 2)
                throw new ArgumentException("The number of classes has to be greater than one.");

            Vector classWeight = this.ClassWeightEstimator.ComputeWeights(this.Classes, y1);

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
            for (int i = 0; i < samples.Length; i++)
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
                {
                    samples[i].Add(new Feature(x.ColumnCount + 1, this.Bias));
                }
            }

            problem.x = samples.Select(s => s.ToArray()).ToArray();
            problem.y = y1.Select(v => (double)v).ToArray();
            problem.l = x.RowCount;
            problem.n = this.Bias > 0 ? x.ColumnCount + 1 : x.ColumnCount;

            Parameter prm = new Parameter(this.solverType, this.C, this.tol);
            prm.weightLabel = Enumerable.Range(0, this.Classes.Length).ToArray();
            prm.weight = classWeight.ToArray();

            this.model = Linear.train(problem, prm);

            //
            int nrClass = model.getNrClass();
            int nrFeature = model.getNrFeature();
            if (Bias > 0)
            {
                nrFeature = nrFeature + 1;
            }

            Matrix<double> r;
            if (nrClass == 2)
            {
                r = DenseMatrix.OfColumnMajor(1, nrFeature, model.getFeatureWeights());
            }
            else
            {
                r = DenseMatrix.OfColumnMajor(nrClass, nrFeature, model.getFeatureWeights());
            }

            if (nrClass > 2)
            {
                var rClone = r.Clone();
                var labels = model.getLabels();
                for (int i = 0; i < labels.Length; i++)
                {
                    rClone.SetRow(labels[i], r.Row(i));
                }

                r = rClone;
            }
            else
            {
                r.MapInplace(v => v*-1);
            }

            if (this.FitIntercept)
            {
                this.Coef = r.SubMatrix(0, r.RowCount, 0, r.ColumnCount - 1);
                this.Intercept = r.Column(r.ColumnCount - 1)*this.interceptScaling;
            }
            else
            {
                this.Coef = r;
                this.Intercept = new DenseVector(r.RowCount);
            }
        }

        /// <summary>
        /// Gets ordered list of class labeled discovered in <see cref="LibLinearBase{TLabel}.Fit"/>.
        /// </summary>
        public TLabel[] Classes
        {
            get { return this._enc.Classes; }
        }

        private double Bias
        {
            get
            {
                if (this.FitIntercept)
                {
                    return this.interceptScaling;
                }
                else
                {
                    return -1.0;
                }
            }
        }
    }
}
