import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

class SolutionStructure:
    """
    A specialized data structure for representing and manipulating collections of chunks.
    
    This structure efficiently tracks:
    - Collections of chunks with no duplicate topics
    - Size ranges and their target proportions
    - Distribution of collections across size ranges
    """
    
    def __init__(self, size_ranges, target_proportions, mode="word"):
        """
        Initialize the solution structure.
        
        Args:
            size_ranges: List of [min_size, max_size] ranges
            target_proportions: List of target proportions for each range
            mode: "word" or "chunk" - determines how collection sizes are measured
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
        
        self.mode = mode  # "word" or "chunk"
        
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
        Add chunks to a specific collection. Will only add all chunks if none of the topics already exist.
        
        Args:
            collection_idx: Index of the collection to add to
            chunks: List of (topic, sentiment, word_count) tuples
        
        Returns:
            True if all chunks were successfully added, False otherwise
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return False
        
        collection = self.collections[collection_idx]
        
        # Pre-validate to ensure no topic already exists in the collection
        for topic, sentiment, word_count in chunks:
            if topic in collection['topics']:
                return False  # If any topic already exists, add nothing
        
        # Calculate old size for range tracking
        old_size = collection['total_word_count'] if self.mode == "word" else collection['total_chunks']
        
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
        new_size = collection['total_word_count'] if self.mode == "word" else collection['total_chunks']
        self._update_size_range_mapping(collection_idx, old_size, new_size)
        
        return True
    
    def remove_chunks_from_collection(self, collection_idx, topics):
        """
        Remove chunks with specified topics from a collection.
        Will only remove all chunks if all topics exist in the collection.
        
        Args:
            collection_idx: Index of the collection to remove from
            topics: List of topics to remove
        
        Returns:
            True if all chunks were successfully removed, False otherwise
        """
        if collection_idx >= len(self.collections) or self.collections[collection_idx] is None:
            return False
        
        collection = self.collections[collection_idx]
        
        # First check if all topics exist in the collection
        for topic in topics:
            if topic not in collection['topics']:
                return False
        
        # Calculate old size for range tracking
        old_size = collection['total_word_count'] if self.mode == "word" else collection['total_chunks']
        
        # Remove all topics
        for topic in topics:
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
        
        # Update average word count
        if collection['total_chunks'] > 0:
            collection['avg_word_count'] = collection['total_word_count'] / collection['total_chunks']
        else:
            collection['avg_word_count'] = 0
        
        # Update size range mapping
        new_size = collection['total_word_count'] if self.mode == "word" else collection['total_chunks']
        self._update_size_range_mapping(collection_idx, old_size, new_size)
        
        return True
    
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
        
        if self.mode == "word":
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
    
    def get_total_absolute_deviation(self):
        """
        Calculate the total absolute deviation from target proportions.
        
        Returns:
            Total absolute deviation as a float
        """
        deviations = self.get_size_range_deviation()
        return sum(abs(dev) for _, dev in deviations)
    
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
        
        if self.mode == "word":
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
        
        if self.mode == "word":
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

    """ VISUALIZATION FUNCTIONS """

    def visualize_solution(self, fig=None, ax=None, title=None, show=True):
        """
        Creates a bar chart visualization of the current solution with vertical range-colored backgrounds.
        
        Args:
            fig: Existing figure to update (or None to create new)
            ax: Existing axis to update (or None to create new)
            title: Optional title for the plot
            show: Whether to show the plot
        
        Returns:
            fig, ax: The figure and axis objects
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        # Clear the axis for updates
        ax.clear()
        
        # Get active collections and sort them by size (descending)
        active_collections = self.get_active_collection_indices()
        active_collections.sort(key=lambda idx: self.get_collection_size(idx), reverse=True)
        
        if not active_collections:
            ax.text(0.5, 0.5, "No collections to display", 
                    horizontalalignment='center', verticalalignment='center')
            if show:
                plt.show()
            return fig, ax
        
        # Vibrant colors for different sentiments
        sentiment_colors = {
            'positive': '#00CC00',  # Bright green
            'neutral': '#808080',   # Medium gray
            'negative': '#FF4500'   # Bright red-orange
        }
        
        # Set up the bars
        collection_positions = np.arange(len(active_collections))
        
        # Define distinct colors for each size range
        range_colors = {
            self.below_min_range_idx: '#FFCCCC',  # Light red for below minimum
            self.above_max_range_idx: '#FFAAAA',  # Darker red for above maximum
        }
        
        # Generate colors for user-defined ranges
        num_user_ranges = len(self.user_size_ranges)
        if num_user_ranges > 0:
            # Create a color map from light blue to dark blue
            blues = plt.cm.Blues(np.linspace(0.3, 0.9, num_user_ranges))
            
            for i in range(num_user_ranges):
                # Map range to the adjusted index (accounting for below_min_range)
                range_colors[i+1] = blues[i]
        
        # Identify the range for each collection
        collection_ranges = [self.get_collection_range_idx(idx) for idx in active_collections]
        
        # Draw vertical background colors for ranges by identifying contiguous groups
        current_range = None
        start_idx = 0
        range_labels = {}  # To track which ranges are present in the chart
        
        for i, range_idx in enumerate(collection_ranges + [None]):  # Add None to handle the last group
            if range_idx != current_range:
                # End of a range group, draw the background if we were tracking a range
                if current_range is not None:
                    color = range_colors.get(current_range, '#EEEEEE')
                    ax.axvspan(start_idx - 0.5, i - 0.5, color=color, alpha=0.2, zorder=0)
                    
                    # Keep track of this range for the legend
                    if current_range == self.below_min_range_idx:
                        range_labels[f"Below min (<{self.global_min})"] = color
                    elif current_range == self.above_max_range_idx:
                        range_labels[f"Above max (>{self.global_max})"] = color
                    else:
                        min_size, max_size = self.size_ranges[current_range]
                        range_labels[f"Range {current_range}: {min_size}-{max_size}"] = color
                    
                # Start new range group
                start_idx = i
                current_range = range_idx
        
        # Plot each collection
        for i, coll_idx in enumerate(active_collections):
            chunks = self.get_all_chunks(coll_idx)
            
            # Sort chunks by size within each collection for better visualization
            chunks_sorted = sorted(chunks, key=lambda x: x[2], reverse=True)
            
            # Draw chunks within collection
            current_height = 0
            for topic, sentiment, word_count in chunks_sorted:
                # Size of the chunk depends on the mode
                chunk_size = word_count if self.mode == "word" else 1
                
                # Draw chunk as a rectangle with only top border
                rect = patches.Rectangle(
                    (i - 0.3, current_height),
                    0.6, 
                    chunk_size,
                    facecolor=sentiment_colors.get(sentiment, '#AAAAAA'), 
                    edgecolor='none',
                    linewidth=0,
                    alpha=0.9
                )
                ax.add_patch(rect)
                
                # Add just a top border line
                ax.plot(
                    [i - 0.3, i + 0.3],  # x points: left and right of bar
                    [current_height, current_height],  # y points: top edge
                    color='black',
                    linewidth=1,
                    solid_capstyle='butt'
                )
                
                # Increment by the chunk size in the current mode
                current_height += chunk_size
        
        # Set labels and title 
        ax.set_xlabel('Collections (sorted by size)')
        ax.set_ylabel(f'{"Word" if self.mode == "word" else "Chunk"} Count')
        ax.set_title(title or f"Solution Visualization ({self.mode}) - {len(active_collections)} Collections")
        
        # Set x-axis ticks
        ax.set_xticks(collection_positions)
        ax.set_xticklabels([f"" for idx in active_collections])
        
        # Create legend elements
        legend_elements = []
        
        # Add sentiment legend patches
        for sent, color in sentiment_colors.items():
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='none', label=f"{sent.capitalize()}")
            )
        
        # Add range legend patches - sort by range index for better readability
        for label, color in sorted(range_labels.items(), key=lambda x: x[0]):
            legend_elements.append(
                patches.Patch(facecolor=color, alpha=0.4, edgecolor='none', label=label)
            )
        
        # Add the legend in a good position
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
        
        # Adjust layout
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, ax