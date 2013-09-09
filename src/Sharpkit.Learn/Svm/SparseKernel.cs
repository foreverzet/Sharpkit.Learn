// -----------------------------------------------------------------------
// <copyright file="SparseKernel.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Generic;

namespace Sharpkit.Learn.Svm
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public enum SparseKernel
    {
        Linear =0,
        Poly =1,
        Rbf =2,
        Sigmoid =3,
        Precomputed =4
    }

    public class Kernel
    {
        public static Kernel FromSparseKernel(SparseKernel kernel)
        {
            return new Kernel { KernelType = kernel};
        }

        public static Kernel FromFunction(Func<Matrix<double>, Matrix<double>, Matrix<double>> f)
        {
            return new Kernel {KernelFunction = f, KernelType = SparseKernel.Precomputed};
        }

        internal SparseKernel KernelType;
        internal Func<Matrix<double>, Matrix<double>, Matrix<double>> KernelFunction;
    }
}
