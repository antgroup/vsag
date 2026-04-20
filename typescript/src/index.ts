import * as path from "path";

const addon = require(path.resolve(__dirname, "../build/Release/vsag_node.node"));

/**
 * Search result containing matched IDs and their distances.
 */
export interface SearchResult {
    ids: BigInt64Array;
    distances: Float32Array;
}

/**
 * Minimum and maximum ID in the index.
 */
export interface MinMaxId {
    minId: number;
    maxId: number;
}

/**
 * Vector search index class for efficient similarity search.
 *
 * Supports dense vector indexing and searching. Provides methods for building
 * indexes, performing k-NN and range searches, and saving/loading indexes to/from disk.
 */
export class Index {
    private _native: any;

    /**
     * Create a new Index instance.
     *
     * @param name - Name of the index type (e.g., "hgraph", "ivf", "sindi")
     * @param parameters - JSON string containing index configuration parameters
     */
    constructor(name: string, parameters: string) {
        this._native = new addon.Index(name, parameters);
    }

    /**
     * Build index from dense float32 vectors.
     *
     * @param vectors - Float32Array with total size numElements * dim
     * @param ids - BigInt64Array with shape (numElements,)
     * @param numElements - Number of vectors in the dataset
     * @param dim - Dimensionality of each vector
     */
    build(
        vectors: Float32Array,
        ids: BigInt64Array,
        numElements: number,
        dim: number
    ): void {
        this._native.build(vectors, ids, numElements, dim);
    }

    /**
     * Add new vectors to the index dynamically.
     *
     * @param vectors - Float32Array with total size numElements * dim
     * @param ids - BigInt64Array with shape (numElements,)
     * @param numElements - Number of vectors to add
     * @param dim - Dimensionality of each vector
     */
    add(
        vectors: Float32Array,
        ids: BigInt64Array,
        numElements: number,
        dim: number
    ): void {
        this._native.add(vectors, ids, numElements, dim);
    }

    /**
     * Remove vectors from the index by their IDs.
     *
     * @param ids - BigInt64Array of IDs to remove
     * @returns Number of vectors successfully removed
     */
    remove(ids: BigInt64Array): number {
        return this._native.remove(ids);
    }

    /**
     * Perform k-nearest neighbors search on a single dense query vector.
     *
     * @param vector - Float32Array representing the query vector
     * @param k - Number of nearest neighbors to retrieve
     * @param parameters - JSON string containing search-specific parameters
     * @returns Search result with matched IDs and distances
     */
    knnSearch(vector: Float32Array, k: number, parameters: string): SearchResult {
        return this._native.knnSearch(vector, k, parameters);
    }

    /**
     * Perform range search to find all vectors within a specified distance threshold.
     *
     * @param vector - Float32Array representing the query vector
     * @param threshold - Maximum distance threshold for inclusion in results
     * @param parameters - JSON string containing search-specific parameters
     * @returns Search result with matched IDs and distances
     */
    rangeSearch(
        vector: Float32Array,
        threshold: number,
        parameters: string
    ): SearchResult {
        return this._native.rangeSearch(vector, threshold, parameters);
    }

    /**
     * Save the built index to a binary file.
     *
     * @param filename - Path to the output file
     */
    save(filename: string): void {
        this._native.save(filename);
    }

    /**
     * Load a previously saved index from a binary file.
     *
     * The Index object must be constructed with the same parameters
     * that were used when the index was originally built and saved.
     *
     * @param filename - Path to the input file
     */
    load(filename: string): void {
        this._native.load(filename);
    }

    /**
     * Get the number of elements in the index.
     */
    getNumElements(): number {
        return this._native.getNumElements();
    }

    /**
     * Get the memory usage of the index in bytes.
     */
    getMemoryUsage(): number {
        return this._native.getMemoryUsage();
    }

    /**
     * Check if a specific ID exists in the index.
     *
     * @param id - The ID to check
     * @returns True if the ID exists, false otherwise
     */
    checkIdExist(id: number): boolean {
        return this._native.checkIdExist(id);
    }

    /**
     * Get the minimum and maximum IDs in the index.
     *
     * @returns Object with minId and maxId, or {minId: -1, maxId: -1} on failure
     */
    getMinMaxId(): MinMaxId {
        return this._native.getMinMaxId();
    }

    /**
     * Calculate distances between a query vector and vectors specified by IDs.
     *
     * @param query - Float32Array representing the query vector
     * @param ids - BigInt64Array of IDs to calculate distances for
     * @returns Float32Array of distances corresponding to each ID
     */
    calDistanceById(query: Float32Array, ids: BigInt64Array): Float32Array {
        return this._native.calDistanceById(query, ids);
    }
}

/**
 * Disable all logging output from the VSAG library.
 */
export function setLoggerOff(): void {
    addon.setLoggerOff();
}

/**
 * Set logger level to INFO.
 */
export function setLoggerInfo(): void {
    addon.setLoggerInfo();
}

/**
 * Set logger level to DEBUG.
 */
export function setLoggerDebug(): void {
    addon.setLoggerDebug();
}
