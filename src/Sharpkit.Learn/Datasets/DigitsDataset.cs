// -----------------------------------------------------------------------
// <copyright file="DigitsDataset.cs" company="Sharpkit.Learn">
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
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Reflection;
    using MathNet.Numerics.Data.Text;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Digits sample dataset.
    /// </summary>
    public class DigitsDataset
    {
        /// <summary>
        /// Gets data to learn;
        /// </summary>
        public Matrix<double> Data { get; private set; }

        /// <summary>
        /// Gets regression target for each sample.
        /// </summary>
        public int[] Target { get; private set; }

        /// <summary>
        /// Gets full description of the dataset.
        /// </summary>
        public string Description { get; private set; }
        
        /// <summary>
        /// Load and return the digits dataset (classification).
        /// Each datapoint is a 8x8 image of a digit.
        /// =================   ==============
        /// Classes                         10
        /// Samples per class             ~180
        /// Samples total                 1797
        /// Dimensionality                  64
        /// Features             integers 0-16
        /// =================   ==============
        /// </summary>
        /// <returns>Instance of <see cref="DigitsDataset"/>.</returns>
        public static DigitsDataset Load()
        {
            var reader = new DelimitedReader { Sparse = false, Delimiter = " " };
            var assembly = Assembly.GetAssembly(typeof(DigitsDataset));
            
            string descr;
            using (var sr = new StreamReader(assembly.GetManifestResourceStream("Sharpkit.Learn.Datasets.Data.digits.rst")))
            {
                descr = sr.ReadToEnd();
            }

            using (var datastream = assembly.GetManifestResourceStream("Sharpkit.Learn.Datasets.Data.digits.csv.gz"))
            {
                var data = reader.ReadMatrix<double>(new GZipStream(datastream, CompressionMode.Decompress));
                return new DigitsDataset
                           {
                               Data = data.SubMatrix(0, data.RowCount, 0, data.ColumnCount - 1),
                               Target = data.Column(data.ColumnCount - 1).ToArray().Cast<int>().ToArray(),
                               Description = descr
                           };
            }
        }
    }
}
