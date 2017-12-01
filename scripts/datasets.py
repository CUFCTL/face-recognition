# Class definitions for several datasets.
import os

class Dataset(object):
	def __init__(self, path, num_entries, num_classes):
		self.PATH = path
		self.NUM_ENTRIES = num_entries
		self.NUM_CLASSES = num_classes

	def get_class_name(self, i):
		raise NotImplementedError("Every Dataset must implement the get_class_name method!")

	def get_class_path(self, path, i):
		raise NotImplementedError("Every Dataset must implement the get_class_path method!")

	def get_class_files(self, path, i):
		raise NotImplementedError("Every Dataset must implement the get_class_files method!")

	def get_dst_filename(self, i, filename):
		raise NotImplementedError("Every Dataset must implement the get_dst_filename method!")

class FERETDataset(Dataset):
	def __init__(self):
		Dataset.__init__(self, "datasets/feret", 11338, 994)
		self._classes = os.listdir(self.PATH)

	def get_class_name(self, i):
		return self._classes[i]

	def get_class_path(self, path, i):
		return os.path.join(path, self.get_class_name(i))

	def get_class_files(self, path, i):
		class_path = self.get_class_path(path, i)
		return os.listdir(class_path)

	def get_dst_filename(self, i, filename):
		return filename

class MNISTDataset(Dataset):
	def __init__(self):
		Dataset.__init__(self, "datasets/mnist", 10000, 10)

	def get_class_name(self, i):
		return str(i)

	def get_class_path(self, path, i):
		return os.path.join(path, self.get_class_name(i))

	def get_class_files(self, path, i):
		class_path = self.get_class_path(path, i)
		return os.listdir(class_path)

	def get_dst_filename(self, i, filename):
		class_name = self.get_class_name(i)
		return "%s_%s" % (class_name, filename)

class ORLDataset(Dataset):
	def __init__(self):
		Dataset.__init__(self, "datasets/orl_faces", 400, 40)

	def get_class_name(self, i):
		return "s%d" % (i + 1)

	def get_class_path(self, path, i):
		return os.path.join(path, self.get_class_name(i))

	def get_class_files(self, path, i):
		class_path = self.get_class_path(path, i)
		return os.listdir(class_path)

	def get_dst_filename(self, i, filename):
		class_name = self.get_class_name(i)
		return "%s_%s" % (class_name, filename)

class GTEXDataset(Dataset):
	def __init__(self, sorted_sub_dirs):
		Dataset.__init__(self, "datasets/GTEx_Data", 8555, 53)
		self.dirs = sorted_sub_dirs

	def get_class_name(self, i):
		return self.dirs[i]

	def get_class_path(self, path, i):
		return os.path.join(path, self.get_class_name(i))

	def get_class_files(self, path, i):
		class_path = self.get_class_path(path, i)
		return os.listdir(class_path)

	def get_dst_filename(self, i, filename):
		class_name = self.get_class_name(i)
		return "%s_%s" % (class_name, filename)

class GTEXDataset30(Dataset):
	def __init__(self, sorted_sub_dirs):
		Dataset.__init__(self, "datasets/GTEx_Data_30", 8555, 30)
		self.dirs = sorted_sub_dirs

	def get_class_name(self, i):
		return self.dirs[i]

	def get_class_path(self, path, i):
		return os.path.join(path, self.get_class_name(i))

	def get_class_files(self, path, i):
		class_path = self.get_class_path(path, i)
		return os.listdir(class_path)

	def get_dst_filename(self, i, filename):
		class_name = self.get_class_name(i)
		return "%s_%s" % (class_name, filename)

class FCTLDataset(Dataset):
	def __init__(self, sub_dirs):
		Dataset.__init__(self, "datasets/fctl", 588, 15)
		self.dirs = sub_dirs

	def get_class_name(self, i):
		return self.dirs[i]

	def get_class_path(self, path, i):
		return os.path.join(path, self.get_class_name(i))

	def get_class_files(self, path, i):
		class_path = self.get_class_path(path, i)
		return os.listdir(class_path)

	def get_dst_filename(self, i, filename):
		class_name = self.get_class_name(i)
		return "%s_%s" % (class_name, filename)
