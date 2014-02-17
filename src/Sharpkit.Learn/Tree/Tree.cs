// -----------------------------------------------------------------------
// <copyright file="Tree.cs" company="Sharpkit.Learn">
//  Authors: Gilles Louppe <g.louppe@gmail.com>
//           Peter Prettenhofer <peter.prettenhofer@gmail.com>
//           Brian Holt <bdholt1@gmail.com>
//           Noel Dawe <noel@dawe.me>
//           Satrajit Gosh <satrajit.ghosh@gmail.com>
//           Lars Buitinck <L.J.Buitinck@uva.nl>
//           Sergey Zyuzin
//  Licence: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Tree
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    /// Struct-of-arrays representation of a binary decision tree.
    ///
    /// The binary tree is represented as a number of parallel arrays. The i-th
    /// element of each array holds information about the node `i`. Node 0 is the
    /// tree's root. You can find a detailed description of all arrays in
    /// `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    /// nodes, resp. In this case the values of nodes of the other type are
    /// arbitrary!
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal class Tree
    {
        internal const uint _TREE_LEAF = 0xFFFFFFFF;
        private const uint _TREE_UNDEFINED = 0xFFFFFFFE;

        /// <summary>
        /// The current capacity (i.e., size) of the arrays.
        /// </summary>
        public int Capacity { get; private set; }

        /// <summary>
        /// The number of nodes (internal nodes + leaves) in the tree.
        /// </summary>
        public uint NodeCount { get; private set; }

        public uint[] NClasses { get; private set; }

        /// <summary>
        /// children_left[i] holds the node id of the left child of node i.
        /// For leaves, children_left[i] == TREE_LEAF. Otherwise,
        /// children_left[i] > i. This child handles the case where
        /// X[:, feature[i]] &lt; = threshold[i].
        /// </summary>
        public uint[] ChildrenLeft { get; private set; }

        /// <summary>
        ///         children_right[i] holds the node id of the right child of node i.
        /// For leaves, children_right[i] == TREE_LEAF. Otherwise,
        /// children_right[i] > i. This child handles the case where
        /// X[:, feature[i]] > threshold[i].
        /// </summary>
        public uint[] ChildrenRight { get; private set; }

        /// <summary>
        /// feature[i] holds the feature to split on, for the internal node i.
        /// </summary>
        public uint[] Feature { get; private set; }

        /// <summary>
        /// impurity[i] holds the impurity (i.e., the value of the splitting
        /// criterion) at node i.
        /// </summary>
        public double[] Impurity { get; private set; }

        /// <summary>
        /// threshold[i] holds the threshold for the internal node i.
        /// </summary>
        public double[] Threshold { get; private set; }

        /// <summary>
        /// Contains the constant prediction value of each node.
        /// </summary>
        public double[] Value { get; private set; }

        /// <summary>
        /// n_samples[i] holds the number of training samples reaching node i.
        /// </summary>
        public uint[] NNodeSamples { get; private set; }

        private readonly int nFeatures;
        private readonly int nOutputs;
        private readonly uint maxNClasses;
        private readonly uint valueStride;
        private SplitterBase splitter;
        private readonly uint maxDepth;
        private readonly uint minSamplesSplit;
        private readonly uint minSamplesLeaf;

        public Tree(
            int nFeatures,
            uint[] nClasses,
            int nOutputs,
            SplitterBase splitter,
            uint maxDepth,
            uint minSamplesSplit,
            uint minSamplesLeaf)
        {
            // Input/Output layout
            this.nFeatures = nFeatures;
            this.nOutputs = nOutputs;
            this.NClasses = new uint[nOutputs];


            this.maxNClasses = nClasses.Max();
            this.valueStride = (uint)this.nOutputs * this.maxNClasses;


            for (uint k = 0; k < nOutputs; k++)
            {
                this.NClasses[k] = nClasses[k];
            }

            // Parameters
            this.splitter = splitter;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.minSamplesLeaf = minSamplesLeaf;

            // Inner structures
            this.NodeCount = 0;
            this.Capacity = 0;
            this.ChildrenLeft = null;
            this.ChildrenRight = null;
            this.Feature = null;
            this.Threshold = null;
            this.Value = null;
            this.Impurity = null;
            this.NNodeSamples = null;
        }

        private T[] Resize<T>(T[] arr, int newLength)
        {
            var tmpNew = new T[newLength];
            if (arr != null)
            {
                Array.Copy(arr, tmpNew, Math.Min(arr.Length, tmpNew.Length));
            }

            return tmpNew;
        }

        /// <summary>
        /// Resize all inner arrays to `capacity`, if `capacity` &lt; 0, then
        ///       double the size of the inner arrays.
        /// </summary>
        /// <param name="capacity"></param>
        public void Resize(int capacity = -1)
        {
            if (capacity == this.Capacity)
            {
                return;
            }

            if (capacity < 0)
            {
                if (this.Capacity <= 0)
                {
                    capacity = 3; // default initial value
                }
                else
                {
                    capacity = 2 * this.Capacity;
                }
            }

            this.Capacity = capacity;

            this.ChildrenLeft = Resize(ChildrenLeft, capacity);
            this.ChildrenRight = Resize(this.ChildrenRight, capacity);
            this.Feature = Resize(this.Feature, capacity);
            this.Threshold = Resize(this.Threshold, capacity);
            this.Value = Resize(this.Value, capacity * (int)this.valueStride);
            this.Impurity = Resize(this.Impurity, capacity);
            this.NNodeSamples = Resize(this.NNodeSamples, capacity);

            // if capacity smaller than node_count, adjust the counter
            if (capacity < this.NodeCount)
            {
                this.NodeCount = (uint)capacity;
            }
        }

        /// <summary>
        /// Add a node to the tree. The new node registers itself as
        ///       the child of its parent. 
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="isLeft"></param>
        /// <param name="isLeaf"></param>
        /// <param name="feature"></param>
        /// <param name="threshold"></param>
        /// <param name="impurity"></param>
        /// <param name="nNodeSamples"></param>
        /// <returns></returns>
        private uint AddNode(
            uint parent,
            bool isLeft,
            bool isLeaf,
            uint feature,
            double threshold,
            double impurity,
            uint nNodeSamples)
        {
            uint nodeId = this.NodeCount;

            if (nodeId >= this.Capacity)
            {
                this.Resize();
            }

            this.Impurity[nodeId] = impurity;
            this.NNodeSamples[nodeId] = nNodeSamples;

            if (parent != _TREE_UNDEFINED)
            {
                if (isLeft)
                {
                    this.ChildrenLeft[parent] = nodeId;
                }
                else
                {
                    this.ChildrenRight[parent] = nodeId;
                }
            }

            if (isLeaf)
            {
                this.ChildrenLeft[nodeId] = _TREE_LEAF;
                this.ChildrenRight[nodeId] = _TREE_LEAF;
                this.Feature[nodeId] = _TREE_UNDEFINED;
                this.Threshold[nodeId] = _TREE_UNDEFINED;
            }
            else
            {
                // children_left and children_right will be set later
                this.Feature[nodeId] = feature;
                this.Threshold[nodeId] = threshold;
            }

            this.NodeCount += 1;

            return nodeId;
        }

        /// <summary>
        /// Build a decision tree from the training set (X, y).
        /// </summary>
        public void build(MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> x,
                          MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> y,
                          MathNet.Numerics.LinearAlgebra.Generic.Vector<double> sampleWeight = null)
        {
            // Prepare data before recursive partitioning

            // Initial capacity
            int initCapacity;

            if (this.maxDepth <= 10)
            {
                initCapacity = (int)Math.Pow(2, (this.maxDepth + 1)) - 1;
            }
            else
            {
                initCapacity = 2047;
            }

            this.Resize(initCapacity);

            // Recursive partition (without actual recursion)
            SplitterBase splitter = this.splitter;
            splitter.init(x, y, sampleWeight == null ? null : sampleWeight.ToArray());

            uint stackNValues = 5;
            uint stackCapacity = 50;
            uint[] stack = new uint[stackCapacity];


            stack[0] = 0; // start
            stack[1] = splitter.n_samples; // end
            stack[2] = 0; // depth
            stack[3] = _TREE_UNDEFINED; // parent
            stack[4] = 0; // is_left

            uint pos = 0;
            uint feature = 0;
            double threshold = 0;
            double impurity = 0;

            while (stackNValues > 0)
            {
                stackNValues -= 5;

                uint start = stack[stackNValues];
                uint end = stack[stackNValues + 1];
                uint depth = stack[stackNValues + 2];
                uint parent = stack[stackNValues + 3];
                bool isLeft = stack[stackNValues + 4] != 0;

                uint nNodeSamples = end - start;
                bool isLeaf = ((depth >= this.maxDepth) ||
                                (nNodeSamples < this.minSamplesSplit) ||
                                (nNodeSamples < 2 * this.minSamplesLeaf));

                splitter.node_reset(start, end, ref impurity);
                isLeaf = isLeaf || (impurity < 1e-7);

                if (!isLeaf)
                {
                    splitter.node_split(ref pos, ref feature, ref threshold);
                    isLeaf = isLeaf || (pos >= end);
                }

                uint nodeId = this.AddNode(parent, isLeft, isLeaf, feature,
                                              threshold, impurity, nNodeSamples);

                if (isLeaf)
                {
                    // Don't store value for internal nodes
                    splitter.node_value(this.Value, nodeId * this.valueStride);
                }
                else
                {
                    if (stackNValues + 10 > stackCapacity)
                    {
                        stackCapacity *= 2;
                        var newStack = new uint[stackCapacity];
                        Array.Copy(stack, newStack, stack.Length);
                        stack = newStack;
                    }

                    // Stack right child
                    stack[stackNValues] = pos;
                    stack[stackNValues + 1] = end;
                    stack[stackNValues + 2] = depth + 1;
                    stack[stackNValues + 3] = nodeId;
                    stack[stackNValues + 4] = 0;
                    stackNValues += 5;

                    // Stack left child
                    stack[stackNValues] = start;
                    stack[stackNValues + 1] = pos;
                    stack[stackNValues + 2] = depth + 1;
                    stack[stackNValues + 3] = nodeId;
                    stack[stackNValues + 4] = 1;
                    stackNValues += 5;
                }
            }

            this.Resize((int)this.NodeCount);
            this.splitter = null; // Release memory
        }

        /// <summary>
        /// Predict target for X.
        /// </summary>
        public MathNet.Numerics.LinearAlgebra.Generic.Matrix<double>[] predict(
            MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> x)
        {
            uint nSamples = (uint)x.RowCount;

            if (nOutputs == 1)
            {
                MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> @out =
                    DenseMatrix.Create((int)nSamples, (int)maxNClasses, (i, j) => 0.0);

                for (int i = 0; i < nSamples; i++)
                {
                    uint nodeId = 0;

                    // While node_id not a leaf
                    while (ChildrenLeft[nodeId] != _TREE_LEAF)
                    {
                        // ... and children_right[node_id] != _TREE_LEAF:
                        if (x[i, (int)Feature[nodeId]] <= Threshold[nodeId])
                        {
                            nodeId = ChildrenLeft[nodeId];
                        }
                        else
                        {
                            nodeId = ChildrenRight[nodeId];
                        }
                    }

                    uint offset = nodeId * valueStride;

                    for (int c = 0; c < NClasses[0]; c++)
                    {
                        @out[i, c] = Value[offset + c];
                    }
                }

                return new[] { @out };
            }
            else // n_outputs > 1
            {
                // out_multi = np.zeros((n_samples,
                //                      n_outputs,
                //                      max_n_classes), dtype=np.float64)
                // Note: I've changed order
                var outMulti = new MathNet.Numerics.LinearAlgebra.Generic.Matrix<double>[nOutputs];
                for (int i = 0; i < outMulti.Length; i++)
                {
                    outMulti[i] = DenseMatrix.Create((int)nSamples, (int)maxNClasses, (c, r) => 0.0);
                }

                for (int i = 0; i < nSamples; i++)
                {
                    uint nodeId = 0;

                    // While node_id not a leaf
                    while (ChildrenLeft[nodeId] != _TREE_LEAF)
                    {
                        // ... and children_right[node_id] != _TREE_LEAF:
                        if (x[i, (int)Feature[nodeId]] <= Threshold[nodeId])
                        {
                            nodeId = ChildrenLeft[nodeId];
                        }
                        else
                        {
                            nodeId = ChildrenRight[nodeId];
                        }
                    }

                    uint offset = nodeId*valueStride;

                    for (int k = 0; k < nOutputs; k++)
                    {
                        for (int c = 0; c < NClasses[k]; c++)
                        {
                            //out_multi[i, k, c] = value[offset + c];
                            //Note: I've changed order
                            outMulti[k][i, c] = Value[offset + c];
                        }
                        offset += maxNClasses;
                    }
                }

                return outMulti;
            }
        }

        /// <summary>
        /// Finds the terminal region (=leaf node) for each sample in X.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public MathNet.Numerics.LinearAlgebra.Generic.Vector<double> apply(
            MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> x)
        {
            //cdef double* threshold = this.threshold

            uint nSamples = (uint)x.RowCount;

            MathNet.Numerics.LinearAlgebra.Generic.Vector<double> @out = DenseVector.Create((int)nSamples, i => 0.0);

            for (int i = 0; i < nSamples; i++)
            {
                uint nodeId = 0;

                // While node_id not a leaf
                while (ChildrenLeft[nodeId] != _TREE_LEAF)
                {
                    // ... and children_right[node_id] != _TREE_LEAF:
                    if (x[i, (int)Feature[nodeId]] <= Threshold[nodeId])
                    {
                        nodeId = ChildrenLeft[nodeId];
                    }
                    else
                    {
                        nodeId = ChildrenRight[nodeId];
                    }
                }

                @out[i] = nodeId;
            }

            return @out;
        }

        /// <summary>
        /// Computes the importance of each feature (aka variable).
        /// </summary>
        /// <param name="normalize"></param>
        public MathNet.Numerics.LinearAlgebra.Generic.Vector<double> ComputeFeatureImportances(bool normalize = true)
        {
            MathNet.Numerics.LinearAlgebra.Generic.Vector<double> importances = 
                DenseVector.Create(this.nFeatures, i => 0.0);

            for (uint node = 0; node < NodeCount; node++)
            {
                if (ChildrenLeft[node] != _TREE_LEAF)
                {
                    // ... and children_right[node] != _TREE_LEAF:
                    uint nLeft = NNodeSamples[ChildrenLeft[node]];
                    uint nRight = NNodeSamples[ChildrenRight[node]];

                    importances[(int)Feature[node]] +=
                        NNodeSamples[node]*Impurity[node]
                        - nLeft*Impurity[ChildrenLeft[node]]
                        - nRight*Impurity[ChildrenRight[node]];
                }
            }

            importances = importances.Divide(this.NNodeSamples[0]);

            if (normalize)
            {
                double normalizer = importances.Sum();

                if (normalizer > 0.0)
                {
                    //   Avoid dividing by zero (e.g., when root is pure)
                    importances /= normalizer;
                }
            }

            return importances;
        }
    }
}
