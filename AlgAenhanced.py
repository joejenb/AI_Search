import os
import copy
import sys
import time
import random
import numpy as np

def read_file_into_string(input_file, from_ord, to_ord):
    # take a file "input_file", read it character by character, strip away all unwanted
    # characters with ord < "from_ord" and ord > "to_ord" and return the concatenation
    # of the file as the string "output_string"
    the_file = open(input_file,'r')
    current_char = the_file.read(1)
    output_string = ""
    while current_char != "":
        if ord(current_char) >= from_ord and ord(current_char) <= to_ord:
            output_string = output_string + current_char
        current_char = the_file.read(1)
    the_file.close()
    return output_string

def stripped_string_to_int(a_string):
    # take a string "a_string" and strip away all non-numeric characters to obtain the string
    # "stripped_string" which is then converted to an integer with this integer returned
    a_string_length = len(a_string)
    stripped_string = "0"
    if a_string_length != 0:
        for i in range(0,a_string_length):
            if ord(a_string[i]) >= 48 and ord(a_string[i]) <= 57:
                stripped_string = stripped_string + a_string[i]
    resulting_int = int(stripped_string)
    return resulting_int

def get_string_between(from_string, to_string, a_string, from_index):
    # look for the first occurrence of "from_string" in "a_string" starting at the index
    # "from_index", and from the end of this occurrence of "from_string", look for the first
    # occurrence of the string "to_string"; set "middle_string" to be the sub-string of "a_string"
    # lying between these two occurrences and "to_index" to be the index immediately after the last
    # character of the occurrence of "to_string" and return both "middle_string" and "to_index"
    middle_string = ""              # "middle_string" and "to_index" play no role in the case of error
    to_index = -1                   # but need to initialized to something as they are returned
    start = a_string.find(from_string,from_index)
    if start == -1:
        flag = "*** error: " + from_string + " doesn't appear"
        #trace_file.write(flag + "\n")
    else:
        start = start + len(from_string)
        end = a_string.find(to_string,start)
        if end == -1:
            flag = "*** error: " + to_string + " doesn't appear"
            #trace_file.write(flag + "\n")
        else:
            middle_string = a_string[start:end]
            to_index = end + len(to_string)
            flag = "good"
    return middle_string,to_index,flag

def string_to_array(a_string, from_index, num_cities):
    # convert the numbers separated by commas in the file-as-a-string "a_string", starting from index "from_index",
    # which should point to the first comma before the first digit, into a two-dimensional array "distances[][]"
    # and return it; note that we have added a comma to "a_string" so as to find the final distance
    # distance_matrix = []
    if from_index >= len(a_string):
        flag = "*** error: the input file doesn't have any city distances"
        #trace_file.write(flag + "\n")
    else:
        row = 0
        column = 1
        row_of_distances = [0]
        flag = "good"
        while flag == "good":
            middle_string, from_index, flag = get_string_between(",", ",", a_string, from_index)
            from_index = from_index - 1         # need to look again for the comma just found
            if flag != "good":
                flag = "*** error: there aren't enough cities"
                # trace_file.write(flag + "\n")
            else:
                distance = stripped_string_to_int(middle_string)
                row_of_distances.append(distance)
                column = column + 1
                if column == num_cities:
                    distance_matrix.append(row_of_distances)
                    row = row + 1
                    if row == num_cities - 1:
                        flag = "finished"
                        row_of_distances = [0]
                        for i in range(0, num_cities - 1):
                            row_of_distances.append(0)
                        distance_matrix.append(row_of_distances)
                    else:
                        row_of_distances = [0]
                        for i in range(0,row):
                            row_of_distances.append(0)
                        column = row + 1
        if flag == "finished":
            flag = "good"
    return flag

def make_distance_matrix_symmetric(num_cities):
    # make the upper triangular matrix "distance_matrix" symmetric;
    # note that there is nothing returned
    for i in range(1,num_cities):
        for j in range(0,i):
            distance_matrix[i][j] = distance_matrix[j][i]

# read input file into string

#######################################################################################################
############ now we read an input file to obtain the number of cities, "num_cities", and a ############
############ symmetric two-dimensional list, "distance_matrix", of city-to-city distances. ############
############ the default input file is given here if none is supplied via a command line   ############
############ execution; it should reside in a folder called "city-files" whether it is     ############
############ supplied internally as the default file or via a command line execution.      ############
############ if your input file does not exist then the program will crash.                ############

input_file = "AISearchfile048.txt"

#######################################################################################################

# you need to worry about the code below until I tell you; that is, do not touch it!

if len(sys.argv) == 1:
    file_string = read_file_into_string("../city-files/" + input_file,44,122)
else:
    input_file = sys.argv[1]
    file_string = read_file_into_string("../city-files/" + input_file,44,122)
file_string = file_string + ","         # we need to add a final comma to find the city distances
                                        # as we look for numbers between commas
print("I'm working with the file " + input_file + ".")
                                        
# get the name of the file

name_of_file,to_index,flag = get_string_between("NAME=", ",", file_string, 0)

if flag == "good":
    print("I have successfully read " + input_file + ".")
    # get the number of cities
    num_cities_string,to_index,flag = get_string_between("SIZE=", ",", file_string, to_index)
    num_cities = stripped_string_to_int(num_cities_string)
else:
    print("***** ERROR: something went wrong when reading " + input_file + ".")
if flag == "good":
    print("There are " + str(num_cities) + " cities.")
    # convert the list of distances into a 2-D array
    distance_matrix = []
    to_index = to_index - 1             # ensure "to_index" points to the comma before the first digit
    flag = string_to_array(file_string, to_index, num_cities)
if flag == "good":
    # if the conversion went well then make the distance matrix symmetric
    make_distance_matrix_symmetric(num_cities)
    print("I have successfully built a symmetric two-dimensional array of city distances.")
else:
    print("***** ERROR: something went wrong when building the two-dimensional array of city distances.")

#######################################################################################################
############ end of code to build the distance matrix from the input file: so now you have ############
############ the two-dimensional "num_cities" x "num_cities" symmetric distance matrix     ############
############ "distance_matrix[][]" where "num_cities" is the number of cities              ############
#######################################################################################################

# now you need to supply some parameters ...

#######################################################################################################
############ YOU NEED TO INCLUDE THE FOLLOWING PARAMETERS:                                 ############
############ "my_user_name" = your user-name, e.g., mine is dcs0ias                        ############

my_user_name = "gltc66"

############ "my_first_name" = your first name, e.g., mine is Iain                         ############

my_first_name = "Jospeh"

############ "my_last_name" = your last name, e.g., mine is Stewart                        ############

my_last_name = "Jenner-Bailey"

############ "alg_code" = the two-digit code that tells me which algorithm you have        ############
############ implemented (see the assignment pdf), where the codes are:                    ############
############    BF = brute-force search                                                    ############
############    BG = basic greedy search                                                   ############
############    BS = best_first search without heuristic data                              ############
############    ID = iterative deepening search                                            ############
############    BH = best_first search with heuristic data                                 ############
############    AS = A* search                                                             ############
############    HC = hilling climbing search                                               ############
############    SA = simulated annealing search                                            ############
############    GA = genetic algorithm                                                     ############

alg_code = "GA"

############ you can also add a note that will be added to the end of the output file if   ############
############ you like, e.g., "in my basic greedy search, I broke ties by always visiting   ############
############ the first nearest city found" or leave it empty if you wish                   ############

added_note = ""

############ the line below sets up a dictionary of codes and search names (you need do    ############
############ nothing unless you implement an alternative algorithm and I give you a code   ############
############ for it when you can add the code and the algorithm to the dictionary)         ############

codes_and_names = {'BF' : 'brute-force search',
                   'BG' : 'basic greedy search',
                   'BS' : 'best_first search without heuristic data',
                   'ID' : 'iterative deepening search',
                   'BH' : 'best_first search with heuristic data',
                   'AS' : 'A* search',
                   'HC' : 'hilling climbing search',
                   'SA' : 'simulated annealing search',
                   'GA' : 'genetic algorithm'}

#######################################################################################################
############    now the code for your algorithm should begin                               ############
#######################################################################################################

pop_size = 200
mut_percentage = 45
threshold = max(map(max, distance_matrix)) * num_cities
elite_size = 100
random_size = pop_size - elite_size

class Gene:
	def __init__(self, fitness=0, path=None):
		self._adj_mat = {}
		self._fitness = threshold - fitness 
		self._path = path

	def calc_adj_mat(self):
		for i in range(num_cities):
			self.adj_mat[self.path[i]] = {self.path[i-1], self.path[(i+1)%num_cities]}

	def mutate(self):
		if random.randrange(1000) <= mut_percentage:
			split_1 = random.randint(0, num_cities)
			split_2 = random.randint(split_1, num_cities)
			self.fitness += distance_matrix[self.path[split_1%num_cities]][self.path[split_1-1]]
			self.fitness += distance_matrix[self.path[split_2%num_cities]][self.path[split_2-1]]
			self.path[split_1:split_2] = self.path[split_1:split_2][::-1]
			self.fitness -= distance_matrix[self.path[split_1%num_cities]][self.path[split_1-1]]
			self.fitness -= distance_matrix[self.path[split_2%num_cities]][self.path[split_2-1]]

	def calc_fitness(self):
		self._fitness = threshold
		for i in range(num_cities-1):
			self._fitness -= distance_matrix[self.path[i]][self.path[i+1]]
		self._fitness -= distance_matrix[self.path[-1]][self.path[0]]

	def gaurentee_mutate(self):
		split_1 = random.randint(0, num_cities)
		split_2 = random.randint(split_1, num_cities)
		self.fitness += distance_matrix[self.path[split_1%num_cities]][self.path[split_1-1]]
		self.fitness += distance_matrix[self.path[split_2%num_cities]][self.path[split_2-1]]
		self.path[split_1:split_2] = self.path[split_1:split_2][::-1]
		self.fitness -= distance_matrix[self.path[split_1%num_cities]][self.path[split_1-1]]
		self.fitness -= distance_matrix[self.path[split_2%num_cities]][self.path[split_2-1]]

	@property
	def adj_mat(self):
		return self._adj_mat

	@adj_mat.setter
	def adj_mat(self, x):
		self._adj_mat = x

	@property
	def fitness(self):
		return self._fitness
	
	@fitness.setter
	def fitness(self, x):
		self._fitness = x

	@property
	def path(self):
		return self._path

	@path.setter
	def path(self, x):
		self._path = x


def nearest_neighbour():
	population, pop_total, best = [], 0, Gene()
	best.fitness = 0
	for start in range(num_cities):
		total, tour = 0, [start]
		unvisited = [t for t in range(num_cities) if t != start]
		for i in range(num_cities-1):
			rem_dis = [(distance_matrix[tour[-1]][x], x) for x in unvisited]
			closest_cost = min(rem_dis, key=lambda x: x[0])
			total += closest_cost[0]
			tour.append(closest_cost[1])
			unvisited.remove(closest_cost[1])
		total += distance_matrix[tour[-1]][tour[0]]
		pop_total += total
		new = Gene(total, tour)
		if total > best.fitness:
			best = new
		population.append(new)
	return population, pop_total, best


def get_min(u_mat, n):
	smallest = min(u_mat[n], key = lambda x: len(u_mat[x]))
	all_small = [el for el in u_mat[n] if len(u_mat[smallest]) == len(u_mat[el])]
	return random.choice(all_small)


def main():
	cities = [i for i in range(num_cities)]
	population, total, max_f = nearest_neighbour() 
	peak_time = 1
	for i in range(pop_size-num_cities):
		par = random.choice(population)
		mem = copy.deepcopy(par)
		mem.gaurentee_mutate()
		
		total += mem.fitness
		population.append(mem)# Try list pairs
	while time.time() - start < 115:# Try while loop
		population, peak_time, max_f, total = evolve(population, peak_time, max_f, total)
	return max_f.path, threshold-max_f.fitness


def evolve(population, peak_time, max_f, total):
	population.sort(key=lambda x: x.fitness, reverse=True) 
	new_pop, new_total = population[:elite_size], 0
	normalised = [x.fitness/total for x in population]# Try mapping
	for i in range(random_size):
		x = population[np.random.choice(pop_size, 1, normalised)[0]]# Try generating both in one go
		y = population[np.random.choice(pop_size, 1, normalised)[0]]
		z = reproduce(x, y)
		if z.fitness > max_f.fitness:
			max_f, peak_time = z, 0
		new_pop.append(z)
		new_total += z.fitness
	peak_time += 1
	return new_pop, peak_time, max_f, new_total


def check_tour(tour):
	check_tour_length = 0
	for i in range(0,num_cities-1):
		check_tour_length = check_tour_length + distance_matrix[tour[i]][tour[i+1]]
	check_tour_length = check_tour_length + distance_matrix[tour[num_cities-1]][tour[0]]
	return check_tour_length


def reproduce(x, y):
	p_1, p_2 = Gene(), Gene()
	p_1.path, p_1.fitness = ERO_crossover(x, y)
	p_2.path, p_2.fitness = ERO_crossover(y, x)
	if p_1.fitness > p_2.fitness:
		p_1.mutate()
		return p_1
	else:
		p_2.mutate()
		return p_2


def ERO_crossover(p1, p2):
	n = p1.path[0]
	k, fit = [n], threshold 
	u_mat = {}
	p1.calc_adj_mat()
	p2.calc_adj_mat()
	for i in range(num_cities):
		u_mat[i] = p1.adj_mat[i].union(p2.adj_mat[i])
	while len(k) < num_cities:
		for each in u_mat:
			u_mat[each].discard(n)

		if u_mat[n]:
			new_n = get_min(u_mat, n)
			del u_mat[n]
		else:
			del u_mat[n]
			new_n = random.choice(list(u_mat.keys()))
		n = new_n
		fit -= distance_matrix[k[-1]][n]
		k.append(n)
	fit -= distance_matrix[k[0]][n]
	return k, fit

start = time.time()
tour, tour_length = main()
#######################################################################################################
############ the code for your algorithm should now be complete and you should have        ############
############ computed a tour held in the list "tour" of length "tour_length"               ############
#######################################################################################################

# you do not need to worry about the code below; that is, do not touch it

#######################################################################################################
############ start of code to verify that the constructed tour and its length are valid    ############
#######################################################################################################

check_tour_length = 0
for i in range(0,num_cities-1):
    check_tour_length = check_tour_length + distance_matrix[tour[i]][tour[i+1]]
check_tour_length = check_tour_length + distance_matrix[tour[num_cities-1]][tour[0]]
flag = "good"
if tour_length != check_tour_length:
    flag = "bad"
if flag == "good":
    print("Great! Your tour-length of " + str(tour_length) + " from your " + codes_and_names[alg_code] + " is valid!")
else:
    print("***** ERROR: Your claimed tour-length of " + str(tour_length) + "is different from the true tour length of " + str(check_tour_length) + ".")

#######################################################################################################
############ start of code to write a valid tour to a text (.txt) file of the correct      ############
############ format; if your tour is not valid then you get an error message on the        ############
############ standard output and the tour is not written to a file                         ############
############                                                                               ############
############ the name of file is "my_user_name" + mon-dat-hr-min-sec (11 characters);      ############
############ for example, dcs0iasSep22105857.txt; if dcs0iasSep22105857.txt already exists ############
############ then it is overwritten                                                        ############
#######################################################################################################

if flag == "good":
    local_time = time.asctime(time.localtime(time.time()))   # return 24-character string in form "Tue Jan 13 10:17:09 2009"
    output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
                                                             # output_file_time = mon + day + hour + min + sec (11 characters)
    output_file_name = my_user_name + output_file_time + ".txt"
    f = open(output_file_name,'w')
    f.write("USER = " + my_user_name + " (" + my_first_name + " " + my_last_name + ")\n")
    f.write("ALGORITHM = " + alg_code + ", FILENAME = " + name_of_file + "\n")
    f.write("NUMBER OF CITIES = " + str(num_cities) + ", TOUR LENGTH = " + str(tour_length) + "\n")
    f.write(str(tour[0]))
    for i in range(1,num_cities):
        f.write("," + str(tour[i]))
    if added_note != "":
        f.write("\nNOTE = " + added_note)
    f.close()
    print("I have successfully written the tour to the output file " + output_file_name + ".")
    
    











    


