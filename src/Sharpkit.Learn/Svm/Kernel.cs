// -----------------------------------------------------------------------
// <copyright file="Kernel.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Svm
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Kernel to be used with Support Vector Machines classifiers.
    /// </summary>
    public class Kernel : IEquatable<Kernel>
    {
        /// <summary>
        /// Gets linear kernel.
        /// </summary>
        public static Kernel Linear
        {
            get { return FromLibSvmKernel(LibSvmKernel.Linear); }
        }

        /// <summary>
        /// Gets polynomial kernel.
        /// </summary>
        public static Kernel Poly
        {
            get { return FromLibSvmKernel(LibSvmKernel.Poly); }
        }

        /// <summary>
        /// Gets rbf kernel.
        /// </summary>
        public static Kernel Rbf
        {
            get { return FromLibSvmKernel(LibSvmKernel.Rbf); }
        }

        /// <summary>
        /// Gets sigmoid kernel.
        /// </summary>
        public static Kernel Sigmoid
        {
            get { return FromLibSvmKernel(LibSvmKernel.Sigmoid); }
        }

        /// <summary>
        /// Gets precomputed kernel.
        /// </summary>
        public static Kernel Precomputed
        {
            get { return FromLibSvmKernel(LibSvmKernel.Precomputed); }
        }

        /// <summary>
        /// Gets or sets LibSvm kernel type.
        /// </summary>
        internal LibSvmKernel LibSvmKernel { get; set; }

        /// <summary>
        /// Gets or sets function which calculates kernel matrix.
        /// </summary>
        internal Func<Matrix<double>, Matrix<double>, Matrix<double>> KernelFunction { get; set; }

        /// <summary>
        /// Creates precomputed kernel.
        /// </summary>
        /// <param name="f">Function used to precompute the kernel matrix.</param>
        /// <returns>Instance of <see cref="Kernel"/>.</returns>
        public static Kernel FromFunction(Func<Matrix<double>, Matrix<double>, Matrix<double>> f)
        {
            return new Kernel { KernelFunction = f, LibSvmKernel = LibSvmKernel.Precomputed };
        }

        /// <summary>
        /// Indicates whether the current object is equal to another object of the same type.
        /// </summary>
        /// <returns>
        /// true if the current object is equal to the <paramref name="other"/> parameter; otherwise, false.
        /// </returns>
        /// <param name="other">An object to compare with this object.</param>
        public bool Equals(Kernel other)
        {
            return this.LibSvmKernel == other.LibSvmKernel && this.KernelFunction == other.KernelFunction;
        }

        /// <summary>
        /// Creates <see cref="Kernel"/> from <see cref="Svm.LibSvmKernel"/>.
        /// </summary>
        /// <param name="kernel">LibSvm kernel.</param>
        /// <returns>Instance of <see cref="Kernel"/>.</returns>
        private static Kernel FromLibSvmKernel(LibSvmKernel kernel)
        {
            return new Kernel { LibSvmKernel = kernel };
        }
    }
}