// -----------------------------------------------------------------------
// <copyright file="DiabetesDataset.cs" company="Sharpkit.Learn">
// # Copyright (c) 2007 David Cournapeau &lt;cournape@gmail.com>
//   2010 Fabian Pedregosa &lt;fabian.pedregosa@inria.fr>
//   2010 Olivier Grisel &lt;olivier.grisel@ensta.org>
//   2013 Sergey Zyuzin &lt;forever.zet@gmail.com>
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Datasets
{
    using System;
    using System.IO.Compression;
    using System.Reflection;
    using MathNet.Numerics.Data.Text;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Diabetes sample dataset.
    /// </summary>
    public class DiabetesDataset
    {
        /// <summary>
        /// Prevents a default instance of the DiabetesDataset class from being created.
        /// </summary>
        private DiabetesDataset()
        {
        }

        /// <summary>
        /// Gets data to learn;
        /// </summary>
        public Matrix<double> Data { get; private set; }

        /// <summary>
        /// Gets regression target for each sample.
        /// </summary>
        public Vector<double> Target { get; private set; }

        /// <summary>
        /// Load and return the diabetes dataset (regression).
        /// ==============      ==================
        /// Samples total       442
        /// Dimensionality      10
        /// Features            real, -.2 &lt; x &lt; .2
        /// Targets             integer 25 - 346
        /// ==============      ==================
        /// </summary>
        /// <returns>
        /// Instance of <see cref="DiabetesDataset"/> class.
        /// </returns>
        public static DiabetesDataset Load()
        {
            var reader = new DelimitedReader {Sparse = false, Delimiter = " "};
            var assembly = Assembly.GetAssembly(typeof(DiabetesDataset));
            using (var datastream = assembly.GetManifestResourceStream("Sharpkit.Learn.Datasets.Data.diabetes_data.csv.gz"))
            using (var targetstream = assembly.GetManifestResourceStream("Sharpkit.Learn.Datasets.Data.diabetes_target.csv.gz"))
            {
                var data = reader.ReadMatrix<double>(new GZipStream(datastream, CompressionMode.Decompress));
                var target = reader.ReadMatrix<double>(new GZipStream(targetstream, CompressionMode.Decompress));
                return new DiabetesDataset { Data = data, Target = target.Column(0) };
            }
        }
    }
}
