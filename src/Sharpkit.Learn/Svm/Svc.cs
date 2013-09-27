// -----------------------------------------------------------------------
// <copyright file="Svc.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Svm
{
    using System;

    /// <summary>
    /// <para>
    /// C-Support Vector Classification.
    /// </para>
    /// <para>
    /// The implementations is a based on libsvm. The fit time complexity
    /// is more than quadratic with the number of samples which makes it hard
    /// to scale to dataset with more than a couple of 10000 samples.
    /// </para>
    /// <para>
    /// The multiclass support is handled according to a one-vs-one scheme.
    /// </para>
    /// <para>
    /// For details on the precise mathematical formulation of the provided
    /// kernel functions and how `gamma`, `coef0` and `degree` affect each,
    /// see the corresponding section in the narrative documentation:
    /// :ref:`svm_kernels`.
    ///</para>
    /// <para>
    /// .. The narrative documentation is available at http://scikit-learn.org/
    /// </para>
    /// </summary>
    /// <example>
    ///     var x = DenseMatrix.OfArray(new[,] { {-1, -1}, {-2, -1}, {1, 1}, {2, 1}});
    ///     var y = new[] { 1, 1, 2, 2 };
    ///     var clf = new Svc&lt;int>();
    ///     clf.Fit(X, y)
    ///     Console.WriteLine(clf.Predict(DenseMatrix.OfArray(new[,] {{-0.8, -1}}));
    ///          { 1 }
    ///
    /// </example>
    /*<remarks>
      <see cref="SVR"/>
       Support Vector Machine for Regression implemented using libsvm.
    
      <see cref="LinearSVC"/>
        Scalable Linear Support Vector Machine for classififcation
        implemented using liblinear. Check the See also section of
        LinearSVC for more comparison element.
    </remarks> */
    public class Svc<TLabel> : SvcBase<TLabel>
    {
        /// <summary>
        /// Initializes a new instance of the Svc class.
        /// </summary>
        /// <param name="c">Penalty parameter C of the error term.</param>
        /// <param name="kernel">Specifies the kernel type to be used in the algorithm.</param>
        /// <param name="degree"> Degree of kernel function.
        /// It is significant only in <see cref="Kernel.Poly"/>.</param>
        /// <param name="gamma">
        /// Kernel coefficient for <see cref="Kernel.Rbf"/>, <see cref="Kernel.Poly"/> and <see cref="Kernel.Sigmoid"/>.
        /// If gamma is 0.0 then 1/nFeatures will be used instead.
        /// </param>
        /// <param name="coef0">
        /// Independent term in kernel function.
        /// It is only significant in <see cref="Kernel.Poly"/> and <see cref="Kernel.Sigmoid"/>.
        /// </param>
        /// <param name="shrinking"> Whether to use the shrinking heuristic.</param>
        /// <param name="probability">
        /// Whether to enable probability estimates. This must be enabled prior
        /// to calling <see cref="SvcBase{TLabel}.PredictProba"/>.
        /// </param>
        /// <param name="tol">Tolerance for stopping criterion.</param>
        /// <param name="cacheSize">Size of the kernel cache (in MB).</param>
        /// <param name="classWeightEstimator"> Set the parameter C of class i to class_weight[i]*C for
        /// Svc. If not given, all classes are supposed to have
        /// weight one. The 'auto' mode uses the values of y to
        /// automatically adjust weights inversely proportional to
        /// class frequencies.</param>
        /// <param name="verbose">Enable verbose output. Note that this setting takes advantage of a
        /// per-process runtime setting in libsvm that, if enabled, may not work
        /// properly in a multithreaded context.</param>
        public Svc(
            double c = 1.0,
            Kernel kernel = null,
            int degree = 3,
            double gamma = 0.0,
            double coef0 = 0.0,
            bool shrinking = true,
            bool probability = false,
            double tol = 1e-3,
            int cacheSize = 200,
            ClassWeightEstimator<TLabel> classWeightEstimator = null,
            bool verbose = false): base(
                LibSvmImpl.c_svc,
                kernel ?? Svm.Kernel.Rbf,
                degree,
                gamma,
                coef0,
                tol,
                c,
                0.0,
                0.0,
                shrinking,
                probability,
                cacheSize,
                classWeightEstimator,
                verbose)
        {
        }
    }
}
