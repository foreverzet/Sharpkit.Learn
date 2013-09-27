// -----------------------------------------------------------------------
// <copyright file="LibSvmBase.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Svm
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using LibSvm;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// <para>
    /// Base class for estimators that use libsvm as backing library
    /// </para>
    /// <para>
    /// This implements support vector machine classification and regression.
    /// </para>
    /// </summary>
    public abstract class LibSvmBase
    {
        /// <summary>
        /// Shape of matrix which was passed to <see cref="Fit"/>.
        /// </summary>
        private Tuple<int, int> fitShape;

        /// <summary>
        /// Matrix which was passed to <see cref="Fit"/>.
        /// </summary>
        private Matrix<double> xFit;

        internal LibSvmBase(
            LibSvmImpl impl,
            Kernel kernel,
            int degree,
            double gamma,
            double coef0,
            double tol,
            double c,
            double nu,
            double epsilon,
            bool shrinking,
            bool probability,
            int cacheSize,
            bool verbose)
        {
            this.Param = new svm_parameter();

            this.Impl = impl;
            this.Kernel = kernel;
            this.Degree = degree;
            this.Gamma = gamma;
            this.Coef0 = coef0;
            this.Tol = tol;
            this.C = c;
            this.Nu = nu;
            this.Epsilon = epsilon;
            this.Shrinking = shrinking;
            this.Probability = probability;
            this.CacheSize = cacheSize;
            this.Verbose = verbose;
        }

        /// <summary>
        /// Gets or sets kernel.
        /// </summary>
        public Kernel Kernel { get; set; }

        /// <summary>
        /// Gets or sets the degree of kernel function.
        /// It is significant only in <see cref="Kernel"/> == <see cref="Svm.Kernel.Poly"/>.
        /// </summary>
        public int Degree
        {
            get { return this.Param.degree; }
            set { this.Param.degree = value; }
        }

        /// <summary>
        /// Gets or sets kernel coefficient for <see cref="Svm.Kernel.Rbf"/>,
        /// <see cref="Svm.Kernel.Poly"/> and <see cref="Svm.Kernel.Sigmoid"/>.
        /// If gamma is 0.0 then 1/nFeatures will be used instead.
        /// </summary>
        public double Gamma { get; set; }

        /// <summary>
        /// Gets or sets independent term in kernel function.
        /// It is only significant in <see cref="Svm.Kernel.Poly"/> and <see cref="Svm.Kernel.Sigmoid"/>.
        /// </summary>
        public double Coef0
        {
            get { return this.Param.coef0; }
            set { this.Param.coef0 = value; }
        }
        
        /// <summary>
        /// Gets or sets tolerance for stopping criterion.
        /// </summary>
        public double Tol
        {
            get { return this.Param.eps; }
            set { this.Param.eps = value; }
        }

        /// <summary>
        /// Gets or sets penalty parameter C of the error term.
        /// </summary>
        public double C
        {
            get { return this.Param.C; }
            set { this.Param.C = value; }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to use the shrinking heuristic.
        /// </summary>
        public bool Shrinking
        {
            get { return this.Param.shrinking != 0; }
            set { this.Param.shrinking = value ? 1 : 0; }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to enable probability estimates.
        /// </summary>
        public bool Probability
        {
            get { return this.Param.probability != 0; }
            set { this.Param.probability = value ? 1 : 0; }
        }

        /// <summary>
        /// Gets or sets size of the kernel cache (in MB).
        /// </summary>
        public int CacheSize
        {
            get { return (int)this.Param.cache_size; }
            set { this.Param.cache_size = value; }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to enable verbose output.
        /// Note that this setting takes advantage of a
        /// per-process runtime setting in libsvm that, if enabled, may not work
        /// properly in a multithreaded context.
        /// </summary>
        public bool Verbose { get; set; }
        
        /// <summary>
        /// Gets or sets number of support vectors in each class.
        /// </summary>
        public int[] NSupport
        {
            get { return this.Model.nSV; }
            set { this.Model.nSV = value; }
        }

        /// <summary>
        /// Gets or sets probability estimates, empty array if <see cref="Probability"/> == false.
        /// </summary>
        public double[] ProbA
        {
            get { return this.Model.probA; }
            set { this.Model.probA = value; }
        }

        /// <summary>
        /// Gets or sets probability estimates, empty array if <see cref="Probability"/> == false.
        /// </summary>
        public double[] ProbB
        {
            get { return this.Model.probB; }
            set { this.Model.probB = value; }
        }

        /// <summary>
        /// Gets or sets labels for different classes (only relevant in classification).
        /// </summary>
        public int[] Label
        {
            get { return this.Model.label; }
            set { this.Model.label = value; }
        }

        /// <summary>
        /// Gets or sets coefficients of support vectors in decision function.
        /// </summary>
        public Matrix<double> DualCoef
        {
            get { return ArrayToMatrix(this.Model.sv_coef); }
            set { this.Model.sv_coef = MatrixToArray(value); }
        }

        /// <summary>
        /// Gets [nSupport] index of support vectors.
        /// </summary>
        public int[] Support
        {
            get { return this.Model.sv_indices.Select(s => s - 1).ToArray(); }
        }

        /// <summary>
        /// Gets support vectors ([nSupport, nFeatures]) (equivalent to X.RowsAt(Support)). Will return an
        /// empty array in the case of precomputed kernel.
        /// </summary>
        public Matrix<double> SupportVectors
        {
            get
            {
                var nVectors = Model.SV.Length;
                var nFeatures = this.fitShape.Item2;
                Matrix<double> m = this.IsSparse ?
                    (Matrix<double>)new SparseMatrix(nVectors, nFeatures) :
                    new DenseMatrix(nVectors, nFeatures);
                
                for (int i = 0; i < nVectors; i++)
                {
                    for (int j = this.Kernel.LibSvmKernel == LibSvmKernel.Precomputed ? 1 : 0; j < Model.SV[i].Length; j++)
                    {
                        m[i, Model.SV[i][j].index] = Model.SV[i][j].value;
                    }
                }

                return m;
            }
        }

        /// <summary>
        /// Gets intercept in decision function
        /// </summary>
        public virtual Vector<double> Intercept
        {
            get { return this.Model.rho.ToDenseVector() * -1; }
        }

        /// <summary>
        /// <para>
        /// Gets weights asigned to the features (coefficients in the primal
        /// problem). This is only available in the case of linear kernel.
        /// [nClass-1, nFeatures]
        /// </para>
        /// <para>
        /// This is readonly property derived from <see cref="DualCoef"/> and <see cref="SupportVectors"/>.
        /// </para>
        ///  </summary>
        public Matrix<double> Coef
        {
            get
            {
                if (this.Kernel.LibSvmKernel != LibSvmKernel.Linear)
                {
                    throw new ArgumentException("coef is only available when using a linear kernel");
                }

                if (this.DualCoef.RowCount == 1)
                {
                    // binary classifier
                    return this.DualCoef * this.SupportVectors * -1;
                }

                // 1vs1 classifier
                var coef = OneVsOneCoef(
                    this.DualCoef,
                    this.NSupport,
                    this.SupportVectors);

                if (coef[0] is SparseVector)
                {
                    return SparseMatrix.OfRows(coef.Length, coef[0].Count, coef);
                }

                return DenseMatrix.OfRows(coef.Length, coef[0].Count, coef);
            }
        }

        internal LibSvmImpl Impl
        {
            get { return (LibSvmImpl)this.Param.svm_type; }
            set { this.Param.svm_type = (int)value; }
        }

        /// <summary>
        /// Gets or sets nu.
        /// </summary>
        internal double Nu
        {
            get { return this.Param.nu; }
            set { this.Param.nu = value; }
        }

        /// <summary>
        /// Gets or sets epsilon.
        /// </summary>
        internal double Epsilon
        {
            get { return this.Param.p; }
            set { this.Param.p = value; }
        }

        internal double[] ClassWeight
        {
            get
            {
                return this.Param.weight;
            }

            set
            {
                this.Param.nr_weight = value.Length;
                this.Param.weight_label = Enumerable.Range(0, value.Length).ToArray();
                this.Param.weight = value;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this class was trained using sparse matrix.
        /// </summary>
        protected bool IsSparse { get; private set; }

        /// <summary>
        /// Gets LibSvm model.
        /// </summary>
        protected svm_model Model { get; private set; }

        /// <summary>
        /// Gets or sets LibSvm parameter class.
        /// </summary>
        private svm_parameter Param { get; set; }

        private bool Pairwise
        {
            get { return this.Kernel.LibSvmKernel == LibSvmKernel.Precomputed || Kernel.KernelFunction != null; }
        }

        /// <summary>
        /// Fit the Svm model according to the given training data.
        /// </summary>
        /// <param name="x">
        /// [nSamples, nFeatures]
        /// Training vectors, where nSamples is the number of samples
        /// and nFeatures is the number of features.</param>
        /// <param name="y">[nSamples]
        /// Target values (class labels in classification, real numbers in
        /// regression)
        /// </param>
        public virtual void Fit(Matrix<double> x, Vector<double> y)
        {
            this.IsSparse = x is SparseMatrix && !this.Pairwise;

            if (this.IsSparse && this.Pairwise)
            {
                throw new ArgumentException(
                    "Sparse precomputed kernels are not supported. " +
                    "Using sparse data and dense kernels is possible " +
                    "by not using the ``sparse`` parameter");
            }

            // input validation
            if (this.Impl != LibSvmImpl.one_class && x.RowCount != y.Count)
            {
                throw new ArgumentException(
                    string.Format(
                        "X and y have incompatible shapes.\n X has {0} samples, but y has {1}.",
                        x.RowCount,
                        y.Count));
            }

            if (this.Kernel.LibSvmKernel == LibSvmKernel.Precomputed &&
                this.Kernel.KernelFunction == null && x.RowCount != x.ColumnCount)
            {
                throw new ArgumentException("X.RowCount should be equal to X.ColumnCount");
            }

            if (this.Verbose)
            {
                Console.WriteLine("[LibSVM]");
            }

            this.FitInternal(x, y);
            this.fitShape = x.Shape();
        }

        /// <summary>
        /// <para>
        /// Perform regression on samples in X.
        /// </para>
        /// <para>
        /// For an one-class model, +1 or -1 is returned.
        /// </para>
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]</param>
        /// <returns>[nSamples]</returns>
        internal Vector<double> predict(Matrix<double> x)
        {
            x = this.ValidateForPredict(x);
            return this.PredictInternal(x);
        }

        /// <summary>
        /// Validation of y and class_weight.
        /// </summary>
        /// <param name="y">Target values.</param>
        protected virtual Vector<double> ValidateTargets(Vector<double> y)
        {
            return y;
        }

        /// <summary>
        /// Return the data transformed by a callable kernel.
        /// </summary>
        /// <param name="x"></param>
        protected Matrix<double> ComputeKernel(Matrix<double> x)
        {
            if (this.Kernel.KernelFunction != null)
            {
                // in the case of precomputed kernel given as a function, we
                // have to compute explicitly the kernel matrix
                var vk = Kernel.KernelFunction(x, this.xFit);
                if (vk is SparseMatrix)
                {
                    vk = DenseMatrix.OfMatrix(vk);
                }

                return vk;
            }

            return x;
        }

        protected Matrix<double> ValidateForPredict(Matrix<double> x)
        {
            if (this.IsSparse && !(x is SparseMatrix))
            {
                x = SparseMatrix.OfMatrix(x);
            }

            if ((x is SparseMatrix) && !this.IsSparse && this.Kernel.KernelFunction == null)
            {
                throw new ArgumentException(
                    string.Format(
                        "cannot use sparse input in {0} trained on dense data",
                        this.GetType().Name));
            }

            int nFeatures = x.ColumnCount;

            if (this.Kernel.LibSvmKernel == LibSvmKernel.Precomputed && this.Kernel.KernelFunction == null)
            {
                if (x.ColumnCount != this.fitShape.Item1)
                {
                    throw new ArgumentException(
                        string.Format(
                            "X.shape[1] = {0} should be equal to {1}, " +
                            "the number of samples at training time",
                             x.ColumnCount,
                             this.fitShape.Item1));
                }
            }
            else if (nFeatures != this.fitShape.Item2)
            {
                throw new ArgumentException(
                    string.Format(
                        "X.shape[1] = {0} should be equal to {1}, " +
                        "the number of features at training time",
                        nFeatures,
                        this.fitShape.Item2));
            }

            return x;
        }

        private static int[] CumSum(int[] a)
        {
            int[] result = new int[a.Length];
            int s = 0;
            for (int i = 0; i < a.Length; i++)
            {
                s += a[i];
                result[i] = s;
            }

            return result;
        }

        /// <summary>
        /// Generate primal coefficients from dual coefficients
        /// for the one-vs-one multi class LibSVM in the case
        /// of a linear kernel.
        /// </summary>
        private static Vector<double>[] OneVsOneCoef(
            Matrix<double> dualCoef,
            int[] nSupport,
            Matrix<double> supportVectors)
        {
            // get 1vs1 weights for all n*(n-1) classifiers.
            // this is somewhat messy.
            // shape of dual_coef_ is nSV * (n_classes -1)
            // see docs for details
            int nClass = dualCoef.RowCount + 1;

            // XXX we could do preallocation of coef but
            // would have to take care in the sparse case
            var coef = new List<Vector<double>>();
            var svLocs = CumSum(new[] { 0 }.Concat(nSupport).ToArray());
            for (int class1 = 0; class1 < nClass; class1++)
            {
                // SVs for class1:
                var sv1 = supportVectors.SubMatrix(
                    svLocs[class1],
                    svLocs[class1 + 1] - svLocs[class1],
                    0,
                    supportVectors.ColumnCount);

                for (int class2 = class1 + 1; class2 < nClass; class2++)
                {
                    // SVs for class1:
                    var sv2 = supportVectors.SubMatrix(
                        svLocs[class2],
                        svLocs[class2 + 1] - svLocs[class2],
                        0,
                        supportVectors.ColumnCount);

                    // dual coef for class1 SVs:
                    var alpha1 = dualCoef.Row(class2 - 1).SubVector(
                        svLocs[class1],
                        svLocs[class1 + 1] - svLocs[class1]);

                    // dual coef for class2 SVs:
                    var alpha2 = dualCoef.Row(class1).SubVector(
                        svLocs[class2],
                        svLocs[class2 + 1] - svLocs[class2]);

                    // build weight for class1 vs class2
                    coef.Add((alpha1 * sv1) + (alpha2 * sv2));
                }
            }

            return coef.ToArray();
        }

        private static double[][] MatrixToArray(Matrix<double> m)
        {
            var arr = new double[m.RowCount][];

            foreach (var row in m.RowEnumerator())
            {
                arr[row.Item1] = new double[row.Item2.Count];
                foreach (var c in row.Item2.GetIndexedEnumerator())
                {
                    arr[row.Item1][c.Item1] = c.Item2;
                }
            }

            return arr;
        }

        private Matrix<double> ArrayToMatrix(double[][] arr)
        {
            Matrix<double> m = this.IsSparse ?
                (Matrix<double>)new SparseMatrix(arr.Length, arr[0].Length) :
                new DenseMatrix(arr.Length, arr[0].Length);

            for (int i = 0; i < arr.Length; i++)
            {
                for (int j = 0; j < arr[i].Length; j++)
                {
                    m[i, j] = arr[i][j];
                }
            }

            return m;
        }

        private Vector<double> PredictInternal(Matrix<double> x)
        {
            x = this.ComputeKernel(x);

            if (this.Kernel.KernelFunction != null)
            {
                if (x.ColumnCount != this.fitShape.Item1)
                {
                    throw new ArgumentException(
                        string.Format(
                            "X.shape[1] ={0} should be equal to {1}, " +
                            "the number of samples at training time",
                            x.ColumnCount,
                            this.fitShape.Item1));
                }
            }

            var result = new DenseVector(x.RowCount);
            foreach (var r in x.RowEnumerator())
            {
                svm_node[] svmNodes;
                if (this.Kernel.LibSvmKernel == LibSvmKernel.Precomputed)
                {
                    svmNodes = r.Item2.Select(
                        (v, i) => new svm_node { index = i + 1, value = v }).ToArray();
                    svmNodes = new[] { new svm_node { index = 0, value = 1 } }.Concat(svmNodes).ToArray();
                }
                else
                {
                    svmNodes = r.Item2.Select((v, i) => new svm_node { index = i, value = v }).ToArray();
                }

                result[r.Item1] = svm.svm_predict(this.Model, svmNodes);
            }

            return result;
        }

        private void FitInternal(Matrix<double> x, Vector<double> y)
        {
            if (this.Kernel.KernelFunction != null)
            {
                // you must store a reference to X to compute the kernel in predict
                // TODO: add keyword copy to copy on demand
                this.xFit = x;
                x = this.ComputeKernel(x);

                if (x.RowCount != x.ColumnCount)
                {
                    throw new ArgumentException("X.RowCount should be equal to X.ColumnCount");
                }
            }

            var problem = new svm_problem();
            problem.l = x.RowCount;
            problem.x = new svm_node[x.RowCount][];
            foreach (var row in x.RowEnumerator())
            {
                if (Kernel.LibSvmKernel == LibSvmKernel.Precomputed)
                {
                    var svmNodes =
                        row.Item2.GetIndexedEnumerator().Select(i =>
                            new svm_node
                            {
                                index = i.Item1 + 1,
                                value = i.Item2
                            });

                    problem.x[row.Item1] =
                        new[]
                            {
                                new svm_node
                                {
                                    index = 0,
                                    value = row.Item1 + 1
                                }
                            }.Concat(svmNodes).ToArray();
                }
                else
                {
                    var svmNodes =
                        row.Item2.GetIndexedEnumerator().Select(
                        i => new svm_node { index = i.Item1, value = i.Item2 });

                    problem.x[row.Item1] = svmNodes.ToArray();
                }
            }

            problem.y = y.ToArray();

            this.Param.kernel_type = (int)this.Kernel.LibSvmKernel;
            if (new[] { LibSvmKernel.Poly, LibSvmKernel.Rbf }.Contains(this.Kernel.LibSvmKernel) &&
                    this.Gamma == 0)
            {
                // if custom gamma is not provided ...
                this.Param.gamma = 1.0 / x.ColumnCount;
            }
            else
            {
                this.Param.gamma = this.Gamma;
            }

            this.Model = svm.svm_train(problem, this.Param);
        }
    }
}
