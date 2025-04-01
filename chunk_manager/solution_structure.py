class SolutionStructure:
    """
    A specialized data structure for representing and manipulating collections of chunks.
    
    This structure efficiently tracks:
    - Collections of chunks with no duplicate topics
    - Size ranges and their target proportions
    - Distribution of collections across size ranges
    """
    
    def __init__(self, size_ranges, target_proportions, mode="word_count"):
        """
        Initialize the solution structure.
        
        Args:
            size_ranges: List of [min_size, max_size] ranges
            target_proportions: List of target proportions for each range
            mode: "word_count" or "chunk_count" - determines how collection sizes are measured
        """
        # Find global min and max from user ranges
        self.global_min = min(r[0] for r in size_ranges) if size_ranges else 1
        self.global_max = max(r[1] for r in size_ranges) if size_ranges else float('inf')
        
        # Add special ranges for out-of-range collections
        below_min_range = [1, self.global_min - 1] if self.global_min > 1 else [0, 0]
        above_max_range = [self.global_max + 1, float('inf')]
        
        # Complete ranges list with special ranges
        self.user_size_ranges = size_ranges.copy()
        self.size_ranges = [below_min_range] + size_ranges + [above_max_range]
        
        # Extend target proportions with zeros for special ranges
        self.user_target_proportions = target_proportions.copy()
        self.target_proportions = [0.0] + target_proportions + [0.0]
        
        self.mode = mode  # "word_count" or "chunk_count"
        
        # Initialize collections list
        self.collections = []
        
        # Track which collections belong to each size range
        self.collections_by_size_range = [[] for _ in range(len(self.size_ranges))]
        
        # Count of collections in each size range for proportion calculations
        self.count_by_size_range = [0] * len(self.size_ranges)
        
        # Track the size range index for each collection
        self.collection_size_range = []
        
        # Store indices for special ranges
        self.below_min_range_idx = 0
        self.above_max_range_idx = len(self.size_ranges) - 1
    
    def create_new_collection(self):
        """
        Create a new empty collection and add it to the solution.
        
        Returns:
            Index of the new collection
        """
        collection = {
            'topics': {},  # Maps topic -> (sentiment, word_count)
            'total_chunks': 0,
            'total_word_count': 0,
            'avg_word_count': 0,
            'chunks_by_size': []  # List of (topic, word_count) tuples, sorted by word_count desc
        }
        
        self.collections.append(collection)
        collection_idx = len(self.collections) - 1
        
        # New collections start with size 0, which doesn't belong to any size range yet
        self.collection_size_range.append(None)
        
        return collection_idx
    
    def add_chunks_to_collection(self, collection_idx, chunks):
        """
        Add chunks to a specific collection.
        
        Args:
            collection_idx: Index of the collection to add to
            chunks: List of (topic, sentiment, word_count) tuples
        
        Returns:
            Number of successfully added chunks
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return 0
        
        collection = self.collections[collection_idx]
        
        # Pre-validate to ensure no topic already exists in the collection
        for topic, sentiment, word_count in chunks:
            if topic in collection['topics']:
                return 0  # If any topic already exists, add nothing
        
        # Calculate old size for range tracking
        old_size = collection['total_word_count'] if self.mode == "word_count" else collection['total_chunks']
        
        # Add all chunks
        for topic, sentiment, word_count in chunks:
            collection['topics'][topic] = (sentiment, word_count)
            collection['total_chunks'] += 1
            collection['total_word_count'] += word_count
            
            # Insert into chunks_by_size maintaining sorted order (descending)
            pos = self._binary_search_position(collection['chunks_by_size'], word_count, key=lambda x: -x[1])
            collection['chunks_by_size'].insert(pos, (topic, word_count))
        
        # Update average word count
        if collection['total_chunks'] > 0:
            collection['avg_word_count'] = collection['total_word_count'] / collection['total_chunks']
        
        # Update size range mapping
        new_size = collection['total_word_count'] if self.mode == "word_count" else collection['total_chunks']
        self._update_size_range_mapping(collection_idx, old_size, new_size)
        
        return len(chunks)
    
    def remove_chunks_from_collection(self, collection_idx, topics):
        """
        Remove chunks with specified topics from a collection.
        
        Args:
            collection_idx: Index of the collection to remove from
            topics: List of topics to remove
        
        Returns:
            Number of successfully removed chunks
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return 0
        
        collection = self.collections[collection_idx]
        
        # Calculate old size for range tracking
        old_size = collection['total_word_count'] if self.mode == "word_count" else collection['total_chunks']
        
        # Count successful removals
        removed_count = 0
        
        for topic in topics:
            if topic in collection['topics']:
                sentiment, word_count = collection['topics'][topic]
                
                # Remove from topics dictionary
                del collection['topics'][topic]
                
                # Update metadata
                collection['total_chunks'] -= 1
                collection['total_word_count'] -= word_count
                
                # Remove from chunks_by_size
                for i, (t, wc) in enumerate(collection['chunks_by_size']):
                    if t == topic:
                        collection['chunks_by_size'].pop(i)
                        break
                
                removed_count += 1
        
        # Update average word count
        if collection['total_chunks'] > 0:
            collection['avg_word_count'] = collection['total_word_count'] / collection['total_chunks']
        else:
            collection['avg_word_count'] = 0
        
        # Update size range mapping
        new_size = collection['total_word_count'] if self.mode == "word_count" else collection['total_chunks']
        self._update_size_range_mapping(collection_idx, old_size, new_size)
        
        return removed_count
    
    def remove_collection(self, collection_idx):
        """
        Remove an entire collection.
        
        Args:
            collection_idx: Index of the collection to remove
        
        Returns:
            True if successful, False otherwise
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return False
        
        # Get the size range this collection belongs to
        size = self.get_collection_size(collection_idx)
        range_idx = self._get_size_range_index(size)
        
        # Remove from size range tracking
        if range_idx is not None:
            self.collections_by_size_range[range_idx].remove(collection_idx)
            self.count_by_size_range[range_idx] -= 1
        
        # Mark the collection as None
        self.collections[collection_idx] = None
        self.collection_size_range[collection_idx] = None
        
        return True
    
    def get_collection_range_idx(self, collection_idx):
        """
        Get the size range that a collection falls into.
        
        Args:
            collection_idx: Index of the collection
        
        Returns:
            A tuple of (min_size, max_size) representing the range, or None if the collection
            doesn't exist or is empty
        """
        if (collection_idx >= len(self.collections) or 
            self.collections[collection_idx] is None or
            self.collection_size_range[collection_idx] is None):
            return None
            
        # Retrieve range index based for the provided collection index
        return self.collection_size_range[collection_idx]
    
    def get_chunk_by_topic(self, collection_idx, topic):
        """
        Get chunk with specific topic from a collection.
        
        Args:
            collection_idx: Index of the collection
            topic: Topic to retrieve
        
        Returns:
            Tuple of (topic, sentiment, word_count) or None if not found
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return None
        
        collection = self.collections[collection_idx]
        
        if topic in collection['topics']:
            sentiment, word_count = collection['topics'][topic]
            return (topic, sentiment, word_count)
        
        return None
    
    def get_chunks_by_size_order(self, collection_idx):
        """
        Get chunks from a collection in descending size order.
        
        Args:
            collection_idx: Index of the collection
        
        Returns:
            List of (topic, sentiment, word_count) tuples sorted by word_count
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return []
        
        collection = self.collections[collection_idx]
        
        # Convert (topic, word_count) to (topic, sentiment, word_count)
        result = []
        for topic, word_count in collection['chunks_by_size']:
            sentiment = collection['topics'][topic][0]
            result.append((topic, sentiment, word_count))
        
        return result
    
    def get_all_chunks(self, collection_idx):
        """
        Get all chunks from a collection.
        
        Args:
            collection_idx: Index of the collection
        
        Returns:
            List of (topic, sentiment, word_count) tuples
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return []
        
        collection = self.collections[collection_idx]
        result = []
        
        for topic, chunk_data in collection['topics'].items():
            sentiment, word_count = chunk_data
            result.append((topic, sentiment, word_count))
        
        return result
    
    def can_add_chunk_to_collection(self, collection_idx, topic):
        """
        Check if a chunk with given topic can be added to a collection.
        
        Args:
            collection_idx: Index of the collection
            topic: Topic to check
        
        Returns:
            True if the chunk can be added, False otherwise
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return False
        
        collection = self.collections[collection_idx]
        return topic not in collection['topics']
    
    def get_collection_size(self, collection_idx):
        """
        Get the size of a collection based on the current mode.
        
        Args:
            collection_idx: Index of the collection
        
        Returns:
            Size as word count or chunk count, depending on mode
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return 0
        
        collection = self.collections[collection_idx]
        
        if self.mode == "word_count":
            return collection['total_word_count']
        else:
            return collection['total_chunks']
        
    def get_collection_avg_word_count(self, collection_idx):
        """
        Get the average word count of a collection.
        
        Args:
            collection_idx: Index of the collection
        
        Returns:
            Average word count, or 0 if the collection is empty
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return 0
        
        collection = self.collections[collection_idx]
        return collection['avg_word_count']
    
    def get_collections_by_size_range(self, range_idx):
        """
        Get indices of collections that fall within a specific size range.
        
        Args:
            range_idx: Index of the size range
        
        Returns:
            List of collection indices
        """
        if range_idx >= len(self.size_ranges):
            return []
        
        return self.collections_by_size_range[range_idx].copy()
    
    def get_active_collection_indices(self):
        """
        Get indices of all active (non-empty) collections.
        
        Returns:
            List of collection indices
        """
        return [i for i, c in enumerate(self.collections) 
                if c is not None and c['total_chunks'] > 0]
    
    def get_total_collections(self):
        """
        Get the total number of non-empty collections.
        
        Returns:
            Total number of collections
        """
        return sum(1 for c in self.collections if c is not None and c['total_chunks'] > 0)
    
    def calculate_size_distribution(self):
        """
        Calculate current distribution of collections across size ranges.
        
        Returns:
            List of proportions for each size range
        """
        total_collections = sum(self.count_by_size_range)
        
        if total_collections == 0:
            return [0] * len(self.size_ranges)
        
        return [count / total_collections for count in self.count_by_size_range]
    
    def get_size_range_deviation(self):
        """
        Calculate the deviation from target proportions for each size range.
        
        Returns:
            List of (range_idx, deviation) tuples sorted by absolute deviation
        """
        current_distribution = self.calculate_size_distribution()
        deviations = []
        
        for i, (current, target) in enumerate(zip(current_distribution, self.target_proportions)):
            deviation = current - target
            deviations.append((i, deviation))
        
        # Sort by absolute deviation (descending)
        deviations.sort(key=lambda x: -abs(x[1]))
        
        return deviations
    
    def get_overpopulated_ranges(self):
        """
        Get size ranges that have more collections than their target proportion.
        
        Returns:
            List of (range_idx, deviation) tuples sorted by deviation (descending)
        """
        deviations = self.get_size_range_deviation()
        overpopulated = [(idx, dev) for idx, dev in deviations if dev > 0]
        overpopulated.sort(key=lambda x: -x[1])  # Sort by deviation (descending)
        return overpopulated
    
    def get_underpopulated_ranges(self):
        """
        Get size ranges that have fewer collections than their target proportion.
        
        Returns:
            List of (range_idx, deviation) tuples sorted by deviation (ascending)
        """
        deviations = self.get_size_range_deviation()
        underpopulated = [(idx, dev) for idx, dev in deviations if dev < 0]
        underpopulated.sort(key=lambda x: x[1])  # Sort by deviation (ascending)
        return underpopulated
    
    def is_collection_in_size_range(self, collection_idx, range_idx):
        """
        Check if a collection falls within a specific size range.
        
        Args:
            collection_idx: Index of the collection
            range_idx: Index of the size range
        
        Returns:
            True if collection falls within the specified range, False otherwise
        """
        if (collection_idx >= len(self.collections) or 
            self.collections[collection_idx] is None or 
            range_idx >= len(self.size_ranges)):
            return False
        
        size = self.get_collection_size(collection_idx)
        min_size, max_size = self.size_ranges[range_idx]
        
        return min_size <= size <= max_size
    
    def would_be_in_size_range_by_adding(self, collection_idx, chunks, range_idx):
        """
        Check if adding chunks to a collection would put it in a specific size range.
        
        Args:
            collection_idx: Index of the collection
            chunks: List of (topic, sentiment, word_count) tuples to hypothetically add
            range_idx: Index of the size range to check
        
        Returns:
            True if the collection would fall in the range, False otherwise
        """
        if (collection_idx >= len(self.collections) or 
            self.collections[collection_idx] is None or 
            range_idx >= len(self.size_ranges)):
            return False
        
        collection = self.collections[collection_idx]
        
        # Calculate hypothetical new size
        new_chunks = len(chunks)
        new_word_count = sum(wc for _, _, wc in chunks)
        
        if self.mode == "word_count":
            new_size = collection['total_word_count'] + new_word_count
        else:
            new_size = collection['total_chunks'] + new_chunks
        
        min_size, max_size = self.size_ranges[range_idx]
        
        return min_size <= new_size <= max_size
    
    def would_be_in_size_range_by_removing(self, collection_idx, topics, range_idx):
        """
        Check if removing topics from a collection would put it in a specific size range.
        
        Args:
            collection_idx: Index of the collection
            topics: List of topics to hypothetically remove
            range_idx: Index of the size range to check
        
        Returns:
            True if the collection would fall in the range, False otherwise
        """
        if (collection_idx >= len(self.collections) or 
            self.collections[collection_idx] is None or 
            range_idx >= len(self.size_ranges)):
            return False
        
        collection = self.collections[collection_idx]
        
        # Calculate hypothetical new size
        removed_word_count = sum(collection['topics'][topic][1] for topic in topics if topic in collection['topics'])
        
        if self.mode == "word_count":
            new_size = collection['total_word_count'] - removed_word_count
        else:
            new_size = collection['total_chunks'] - len(topics)
        
        min_size, max_size = self.size_ranges[range_idx]
        
        return min_size <= new_size <= max_size
    
    def get_out_of_range_collections(self):
        """
        Get collections that fall outside the user-defined size ranges.
        
        Returns:
            List of collection indices that are in the special ranges
        """
        below_range = self.collections_by_size_range[self.below_min_range_idx].copy()
        above_range = self.collections_by_size_range[self.above_max_range_idx].copy()
        return below_range + above_range
    
    def get_out_of_range_collections_fraction(self):
        """
        Calculate what fraction of all collections are outside the user-defined ranges.
        
        Returns:
            Float representing the fraction of out-of-range collections
        """
        total_collections = sum(self.count_by_size_range)
        if total_collections == 0:
            return 0.0
            
        out_of_range_count = (self.count_by_size_range[self.below_min_range_idx] + 
                             self.count_by_size_range[self.above_max_range_idx])
        
        return out_of_range_count / total_collections
    
    def _binary_search_position(self, sorted_list, value, key=lambda x: x):
        """
        Binary search to find insertion position in a list.
        
        Args:
            sorted_list: The sorted list to search in
            value: The value to insert
            key: Function to extract the comparison key
        
        Returns:
            Position where the value should be inserted
        """
        left, right = 0, len(sorted_list)
        while left < right:
            mid = (left + right) // 2
            if key(sorted_list[mid]) > value:
                left = mid + 1
            else:
                right = mid
        return left
    
    def _get_size_range_index(self, size):
        """
        Determine which size range a given size falls into.
        
        Args:
            size: The size to check (word count or chunk count)
        
        Returns:
            Index of the matching size range, or None if no range matches
        """
        if size == 0:
            return None  # Empty collections aren't in any range
        
        # Check if size is below min range
        if size < self.global_min:  # Below the first user range
            return self.below_min_range_idx
        
        # Check if size is above max range
        if size > self.global_max:  # Above the last user range
            return self.above_max_range_idx
        
        # Check user-defined ranges (skip special ranges at positions 0 and -1)
        for i in range(1, len(self.size_ranges) - 1):
            min_size, max_size = self.size_ranges[i]
            if min_size <= size <= max_size:
                return i
        
        # This should not happen with properly defined ranges
        return None
    
    def _update_size_range_mapping(self, collection_idx, old_size, new_size):
        """
        Update size range mappings when a collection's size changes.
        
        Args:
            collection_idx: Index of the collection
            old_size: Previous size of the collection
            new_size: New size of the collection
        """
        old_range_idx = self._get_size_range_index(old_size)
        new_range_idx = self._get_size_range_index(new_size)
        
        # If the size range hasn't changed, no update needed
        if old_range_idx == new_range_idx:
            return
        
        # Remove from old range if it existed
        if old_range_idx is not None:
            self.collections_by_size_range[old_range_idx].remove(collection_idx)
            self.count_by_size_range[old_range_idx] -= 1
        
        # Add to new range if it exists
        if new_range_idx is not None:
            self.collections_by_size_range[new_range_idx].append(collection_idx)
            self.count_by_size_range[new_range_idx] += 1
        
        # Update the collection's size range
        self.collection_size_range[collection_idx] = new_range_idx
