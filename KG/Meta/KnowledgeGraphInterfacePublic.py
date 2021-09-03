import requests
import time
import json

class CLOCQ:
	def __init__(self, host="http://localhost", port="7777"):
		self.host = host
		self.port = port
		self.req = requests.Session()

	'''
	translates the KG item id (e.g. Q47774) to the corresponding
	label (e.g. France national association football team)
	'''
	def item_to_label(self, item):
		res = self.req.post(self.host+':'+self.port+'/item_to_label', json = {'item': item})
		json_string = res.content.decode('utf-8')
		labels = json.loads(json_string)
		first_label = labels[0]
		return first_label

	'''
	translates the KG item id (e.g. Q47774) to the corresponding
	aliases in the KG (e.g. France, France national association soccer team, ...)
	'''
	def item_to_aliases(self, item):
		res = self.req.post(self.host+':'+self.port+'/item_to_aliases', json = {'item': item})
		json_string = res.content.decode('utf-8')
		aliases = json.loads(json_string)
		return aliases

	'''
	retrieves the description of the given item, that can be seen on top of wikidata pages
	'''
	def get_description(self, item):
		res = self.req.post(self.host+':'+self.port+'/item_to_description', json = {'item': item})
		json_string = res.content.decode('utf-8')
		aliases = json.loads(json_string)
		return aliases

	'''
	returns the types of the item as: list of (wikidataID, label)-pairs
	'''
	def get_types(self, item):
		res = self.req.post(self.host+':'+self.port+'/type', json = {'item': item})
		json_string = res.content.decode('utf-8')
		types = json.loads(json_string)
		return types

	'''
	a list of two frequencies of the item is returned:
	- number of facts with the item occuring as subject
	- number of facts with the item occuring as object/qualifier-object
	'''
	def frequency(self, item):
		res = self.req.post(self.host+':'+self.port+'/frequency', json = {'item': item})
		json_string = res.content.decode('utf-8')
		frequencies = json.loads(json_string)
		return frequencies

	'''
	returns a list of facts including the item (the 1-hop neighborhood)
	each fact is a n-tuple, with subject, predicate, object and an unspecified number of qualifiers
	'''
	def get_neighborhood(self, item, include_labels, p=10000):
		res = self.req.post(self.host+':'+self.port+'/neighborhood', json = {'item_id': item, 'include_labels': include_labels, 'p': p})
		json_string = res.content.decode('utf-8')
		neighbors = json.loads(json_string)
		return neighbors

	'''
	returns a list of facts in the 2-hop neighborhood of the item
	each fact is a n-tuple, with subject, predicate, object and an unspecified number of qualifiers
	'''
	def get_neighborhood_two_hop(self, item, include_labels, p=10000):
		res = self.req.post(self.host+':'+self.port+'/two_hop_neighborhood', json = {'item_id': item, 'include_labels': include_labels, 'p': p})
		json_string = res.content.decode('utf-8')
		neighbors = json.loads(json_string)
		return neighbors

	'''
	returns a list of paths between item1 and item2. Each path is given by either 1 fact (1-hop connection)
	or 2 facts (2-hop connections)
	'''
	def connect(self, item1, item2):
		res = self.req.post(self.host+':'+self.port+'/connect', json = {'item1': item1, 'item2': item2})
		json_string = res.content.decode('utf-8')
		paths = json.loads(json_string)
		return paths

	'''
	returns the distance of the two items in the graph, given a fact-based definition
	returns 1 if the items are within 1 hop of each other,
	returns 0.5 if the items are within 2 hops of each other,
	and returns 0 otherwise
	'''
	def connectivity_check(self, item1, item2):
		res = self.req.post(self.host+':'+self.port+'/connectivity_check', json = {'item1': item1, 'item2': item2})
		connectivity = float(res.content)
		return connectivity

	'''
	extract a question-specific context for the given question using the CLOCQ algorithm. 
	Returns k (context tuple, context graph)-pairs for the given questions,
	i.e. a mapping of question words to KG items and a question-relevant KG subset.
	'''
	def context(self, question, k=1, p=10000, hyperparameters=(0.3, 0.4, 0.2, 0.1, 20)):
		res = self.req.post(self.host+':'+self.port+'/clocq', json = {'question': question, 'k': k, 'p': p, 'hyperparameters': hyperparameters})
		json_string = res.content.decode('utf-8')
		top_k_contexts = json.loads(json_string)
		return top_k_contexts

'''
MAIN
'''
if __name__ == "__main__":
	clocq = CLOCQ()

	item = 'Q47774'
	
	res = clocq.frequency(item)
	print(res)

	res = clocq.get_description(item)
	print(res)
