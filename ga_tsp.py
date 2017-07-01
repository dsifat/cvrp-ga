import random
import copy
import os
import time
import math
import csv

from tkinter import *
from tkinter.ttk import *

list_of_cities =[]

k_mut_prob = 0.4

k_n_generations = 200

k_population_size = 200

tournament_size =3

elitism = True


csv_cities = True

#vehicle_no=input("For which Vehicle do you want to optimize route?:")
#tmp_vhcl=str(int(vehicle_no)-1)

csv_name = ("inputs/cities.csv")
#files to be written
op_file=open("output.txt","w+")
op_file.write("No. of Generation"+"\t"+"Current Best Length"+"\n")

class City(object):
    def __init__(self, name, x, y):
        # Name and coordinates:
        self.name = name
        self.x = self.graph_x = x
        self.y = self.graph_y = y
        # Appends itself to the global list of cities:
        list_of_cities.append(self)
        # Creates a dictionary of the distances to all the other cities (has to use a value so uses itself - always 0)
        self.distance_to = {self.name:0.0}

    def calculate_distances(self): 
        for city in list_of_cities:
            tmp_dist = City.point_dist(self.x, self.y, city.x, city.y)
            self.distance_to[city.name] = tmp_dist

    # Calculates the distance between two cartesian points..
    def point_dist(x1,y1,x2,y2):
        return ((x1-x2)**2 + (y1-y2)**2)**(1/2)


# Route Class
class Route(object):
    def __init__(self):
        # initiates a route attribute equal to a randomly shuffled list_of_cities
        self.route = sorted(list_of_cities, key=lambda *args: random.random())
        ### Calculates its length:
        self.recalc_rt_len()

    def recalc_rt_len(self):
        # Zeroes its length
        self.length = 0.0
        # for every city in its route attribute:
        for city in self.route:
            # set up a next city variable that points to the next city in the list 
            # and wraps around at the end:
            next_city = self.route[self.route.index(city)-len(self.route)+1]
            # Uses the first city's distance_to attribute to find the distance to the next city:
            dist_to_next = city.distance_to[next_city.name]
            # adds this length to its length attr.
            self.length += dist_to_next

    def pr_cits_in_rt(self):
        cities_str = ''
        for city in self.route:
            cities_str += city.name + ','
        cities_str = cities_str[:-1] # chops off last comma
        print('    ' + cities_str)
        return cities_str

    def pr_vrb_cits_in_rt(self):
        cities_str = '|'
        for city in self.route:
            cities_str += str(city.x) + ',' + str(city.y) + '|'
        print(cities_str)

    def is_valid_route(self):
        for city in list_of_cities:
            # helper function defined up to
            if Route.count_mult(self.route,lambda c: c.name == city.name) > 1:
                return False
        return True

    # Returns the number of pred in sequence (duplicate checking.)
    def count_mult(seq, pred):
        return sum(1 for v in seq if pred(v))


# Contains a population of Route() objects
class RoutePop(object):
    
    def __init__(self, size, initialise):
        self.rt_pop = []
        self.size = size
        # If we want to initialise a population.rt_pop:
        if initialise:
            for x in range(0,size):
                new_rt = Route()
                self.rt_pop.append(new_rt)
            self.get_fittest()

    def get_fittest(self):
        # sorts the list based on the routes' lengths
        sorted_list = sorted(self.rt_pop, key=lambda x: x.length, reverse=False)
        self.fittest = sorted_list[0]
        return self.fittest


# Class for bringing together all of the methods to do with the Genetic Algorithm
class GA(object):
    def crossover_experimental(routeA,routeB):
        # new child Route()
        child_rt = Route()


        # prevents from recalculating
        routeB_len = len(routeB.route)

        # Chooses a random city
        random_city = random.choice(list_of_cities)

        # routeA going down
        # routeB going up

        incrementing_a = True
        incrementing_b = True

        idx_a = routeA.route.index(random_city)
        idx_b = routeB.route.index(random_city)

        idx_a -= 1
        idx_b += 1

        if idx_a < 0:
            incrementing_a = False

        if idx_b >= routeB_len:
            incrementing_b = False

        child_rt.route = [random_city]

        # print(random_city.name)

        while (incrementing_a and incrementing_b):
            # print('idx_a: {}'.format(idx_a))

            if idx_a >= 0:
                if not (routeA.route[idx_a] in child_rt.route):
                    child_rt.route.insert(0, routeA.route[idx_a])

            idx_a -= 1

            if idx_a < 0:
                incrementing_a = False
                break

            # child_rt.pr_cits_in_rt()


            if idx_b < routeB_len:
                if not (routeB.route[idx_b] in child_rt.route):
                    child_rt.route.append(routeB.route[idx_b])

            idx_b += 1

            if idx_b >= routeB_len:
                incrementing_b = False
                break

            # print('idx_b: {}'.format(idx_b))
            # child_rt.pr_cits_in_rt()

        # now either incrementing_a or incementing_b must be false

        shuffled_cities = sorted(routeA.route, key=lambda *args: random.random())
        for city in shuffled_cities:
            if not city in child_rt.route:
                child_rt.route.append(city)

        return child_rt

    def crossover(parent1, parent2):



        # new child Route()
        child_rt = Route()

        for x in range(0,len(child_rt.route)):
            child_rt.route[x] = None

        # Two random integer indices of the parent1:
        start_pos = random.randint(0,len(parent1.route))
        end_pos = random.randint(0,len(parent1.route))


        #### takes the sub-route from parent one and sticks it in itself:
        # if the start position is before the end:
        if start_pos < end_pos:
            # do it in the start-->end order
            for x in range(start_pos,end_pos):
                child_rt.route[x] = parent1.route[x] # set the values to eachother
        # if the start position is after the end:
        elif start_pos > end_pos:
            # do it in the end-->start order
            for i in range(end_pos,start_pos):
                child_rt.route[i] = parent1.route[i] # set the values to eachother


        # Cycles through the parent2. And fills in the child_rt
        # cycles through length of parent2:
        for i in range(len(parent2.route)):
            # if parent2 has a city that the child doesn't have yet:
            if not parent2.route[i] in child_rt.route:
                # it puts it in the first 'None' spot and breaks out of the loop.
                for x in range(len(child_rt.route)):
                    if child_rt.route[x] == None:
                        child_rt.route[x] = parent2.route[i]
                        break
        # repeated until all the cities are in the child route

        # returns the child route (of type Route())
        child_rt.recalc_rt_len()
        return child_rt

    def mutate(route_to_mut):
        # k_mut_prob %
        if random.random() < k_mut_prob:

            # two random indices:
            mut_pos1 = random.randint(0,len(route_to_mut.route)-1)
            mut_pos2 = random.randint(0,len(route_to_mut.route)-1)

            # if they're the same, skip to the chase
            if mut_pos1 == mut_pos2:
                return route_to_mut

            # Otherwise swap them:
            city1 = route_to_mut.route[mut_pos1]
            city2 = route_to_mut.route[mut_pos2]

            route_to_mut.route[mut_pos2] = city1
            route_to_mut.route[mut_pos1] = city2

        # Recalculate the length of the route (updates it's .length)
        route_to_mut.recalc_rt_len()

        return route_to_mut

    def mutate_2opt(route_to_mut):
        # k_mut_prob %
        if random.random() < k_mut_prob:

            for i in range(len(route_to_mut.route)):
                for ii in range(len(route_to_mut.route)): # i is a, i + 1 is b, ii is c, ii+1 is d
                    if (route_to_mut.route[i].distance_to[route_to_mut.route[i-len(route_to_mut.route)+1].name]
                     + route_to_mut.route[ii].distance_to[route_to_mut.route[ii-len(route_to_mut.route)+1].name]
                     > route_to_mut.route[i].distance_to[route_to_mut.route[ii].name]
                     + route_to_mut.route[i-len(route_to_mut.route)+1].distance_to[route_to_mut.route[ii-len(route_to_mut.route)+1].name]):

                        c_to_swap = route_to_mut.route[ii]
                        b_to_swap = route_to_mut.route[i-len(route_to_mut.route)+1]

                        route_to_mut.route[i-len(route_to_mut.route)+1] = c_to_swap
                        route_to_mut.route[ii] = b_to_swap 

            route_to_mut.recalc_rt_len()

        return route_to_mut

    def tournament_select(population):

        # New smaller population (not intialised)
        tournament_pop = RoutePop(size=tournament_size,initialise=False)

        # fills it with random individuals (can choose same twice)
        for i in range(tournament_size-1):
            tournament_pop.rt_pop.append(random.choice(population.rt_pop))
        
        # returns the fittest:
        return tournament_pop.get_fittest()

    def evolve_population(init_pop):

        #makes a new population:
        descendant_pop = RoutePop(size=init_pop.size, initialise=True)

        # Elitism offset (amount of Routes() carried over to new population)
        elitismOffset = 0

        # if we have elitism, set the first of the new population to the fittest of the old
        if elitism:
            descendant_pop.rt_pop[0] = init_pop.fittest
            elitismOffset = 1

        # Goes through the new population and fills it with the child of two tournament winners from the previous populatio
        for x in range(elitismOffset,descendant_pop.size):
            # two parents:
            tournament_parent1 = GA.tournament_select(init_pop)
            tournament_parent2 = GA.tournament_select(init_pop)

            # A child:
            tournament_child = GA.crossover(tournament_parent1, tournament_parent2)

            # Fill the population up with children
            descendant_pop.rt_pop[x] = tournament_child

        # Mutates all the routes (mutation with happen with a prob p = k_mut_prob)
        for route in descendant_pop.rt_pop:
            if random.random() < 0.3:
                GA.mutate(route)

        # Update the fittest route:
        descendant_pop.get_fittest()

        return descendant_pop






class App(object):
    def __init__(self,n_generations,pop_size):
        
        
        if csv_cities:
            self.read_csv()

        self.n_generations = n_generations
        self.pop_size = pop_size

        for city in list_of_cities:
            city.calculate_distances()

        self.set_city_gcoords()
        
        # Initiates a window object & sets its title
        self.window = Tk()
        self.window.wm_title("Generation 0")

        # initiates two canvases, one for current and one for best
        self.canvas_current = Canvas(self.window, height=400, width=400)
        self.canvas_best = Canvas(self.window, height=400, width=400)

        # Initiates two labels
        self.canvas_current_title = Label(self.window, text="Best route of current gen:")
        self.canvas_best_title = Label(self.window, text="Overall best so far:")

        # Initiates a status bar with a string
        self.stat_tk_txt = StringVar()
        self.status_label = Label(self.window, textvariable=self.stat_tk_txt, relief=SUNKEN, anchor=W)

        # creates dots for the cities on both of the canvases
        for city in list_of_cities:
            self.canvas_current.create_oval(city.graph_x-2, city.graph_y-2, city.graph_x + 2, city.graph_y + 2, fill='blue')
            self.canvas_best.create_oval(city.graph_x-2, city.graph_y-2, city.graph_x + 2, city.graph_y + 2, fill='blue')

        # Packs all the widgets (physically creates them and places them in order)
        self.canvas_current_title.pack()
        self.canvas_current.pack()
        self.canvas_best_title.pack()
        self.canvas_best.pack()
        self.status_label.pack(side=BOTTOM, fill=X)

        # Runs the main window loop
        self.window_loop()
        #files to be written
        

    def set_city_gcoords(self):

        # defines some variables (we will set them next)
        min_x = 100000
        max_x = -100000
        min_y = 100000
        max_y = -100000

        #finds the proper maximum/minimum
        for city in list_of_cities:

            if city.x < min_x:
                min_x = city.x
            if city.x > max_x:
                max_x = city.x

            if city.y < min_y:
                min_y = city.y
            if city.y > max_y:
                max_y = city.y

        # shifts the graph_x so the leftmost city starts at x=0, same for y.
        for city in list_of_cities:
            city.graph_x = (city.graph_x + (-1*min_x))
            city.graph_y = (city.graph_y + (-1*min_y))

        # resets the variables now we've made changes
        min_x = 100000
        max_x = -100000
        min_y = 100000
        max_y = -100000

        #finds the proper maximum/minimum
        for city in list_of_cities:

            if city.graph_x < min_x:
                min_x = city.graph_x
            if city.graph_x > max_x:
                max_x = city.graph_x

            if city.graph_y < min_y:
                min_y = city.graph_y
            if city.graph_y > max_y:
                max_y = city.graph_y

        # if x is the longer dimension, set the stretch factor to 300 (px) / max_x. Else do it for y. This conserves aspect ratio.
        if max_x > max_y:
            stretch = 300 / max_x
        else:
            stretch = 300 / max_y

        # stretch all the cities so that the city with the highest coordinates has both x and y < 300
        for city in list_of_cities:
            city.graph_x *= stretch
            city.graph_y = 300 - (city.graph_y * stretch)


    def update_canvas(self,the_canvas,the_route,color):

        # deletes all current items with tag 'path'
        the_canvas.delete('path')

        # loops through the route
        for i in range(len(the_route.route)):

            # similar to i+1 but will loop around at the end
            next_i = i-len(the_route.route)+1

            # creates the line from city to city
            the_canvas.create_line(the_route.route[i].graph_x,
                                the_route.route[i].graph_y,
                                the_route.route[next_i].graph_x,
                                the_route.route[next_i].graph_y,
                                tags=("path"),
                                fill=color)

            # Packs and updates the canvas
            the_canvas.pack()
            the_canvas.update_idletasks()

    def read_csv(self):
        with open(csv_name, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                new_city = City(row[0],float(row[1]),float(row[2]))

    def GA_loop(self,n_generations,pop_size):

        # takes the time to measure the elapsed time
        start_time = time.time()

        # Creates the population:
        the_population = RoutePop(pop_size, True)

        # the_population.rt_pop[0].route = [1,8,38,31,44,18,7,28,6,37,19,27,17,43,30,36,46,33,20,47,21,32,39,48,5,42,24,10,45,35,4,26,2,29,34,41,16,22,3,23,14,25,13,11,12,15,40,9]
        # the_population.rt_pop[0].recalc_rt_len()
        # the_population.get_fittest()

        #checks to make sure there are no duplicate cities:
        if the_population.fittest.is_valid_route() == False:
            raise NameError('Multiple cities with same name. Check cities.')
            return # if there are, raise a NameError and return

        # gets the best length from the first population (no practical use, just out of interest to see improvements)
        initial_length = the_population.fittest.length

        # Creates a random route called best_route. It will store our overall best route.
        best_route = Route()

        # Update the two canvases with the just-created routes:
        self.update_canvas(self.canvas_current,the_population.fittest,'red')
        self.update_canvas(self.canvas_best,best_route,'green')


        # Main process loop (for number of generations)
        for x in range(1,n_generations):
            # Updates the current canvas every n generations (to avoid it lagging out, increase n)
            if x % 8 == 0:
                self.update_canvas(self.canvas_current,the_population.fittest,'red')

            # Evolves the population:
            the_population = GA.evolve_population(the_population)

            # If we have found a new shorter route, save it to best_route
            if the_population.fittest.length < best_route.length:
                # set the route (copy.deepcopy because the_population.fittest is persistent in this loop so will cause reference bugs)
                best_route = copy.deepcopy(the_population.fittest)
                # Update the second canvas because we have a new best route:
                self.update_canvas(self.canvas_best,best_route,'green')
                # update the status bar (bottom bar)
                self.stat_tk_txt.set('Initial length {0:.2f} Best length = {1:.2f}'.format(initial_length,best_route.length))
                self.status_label.pack()
                self.status_label.update_idletasks()
            op_file.write(str(x)+"\t"+str(best_route.length)+"\n")
            # Prints info to the terminal:
            App.clear_term()
            print('Generation {0} of {1}'.format(x,n_generations))
            print(' ')
            print('Overall fittest has length {0:.2f}'.format(best_route.length))
            print('and goes via:')
            best_route.pr_cits_in_rt()
            print(' ')
            print('Current fittest has length {0:.2f}'.format(the_population.fittest.length))
            print('And goes via:')
            the_population.fittest.pr_cits_in_rt()
            print(' ')
            print('''The screen with the maps may become unresponsive if 
the population size is too large. It will refresh at the end.''')
           
            
                

            # sets the window title to the latest Generation:
            self.window.wm_title("Generation {0}".format(x))

        # sets the window title to the last generation
        self.window.wm_title("Generation {0}".format(n_generations))

        # updates the best route canvas for the last time:
        self.update_canvas(self.canvas_best,best_route,'green')
        
        # takes the end time of the run:
        end_time = time.time()

        # Prints final output to terminal:
        App.clear_term()
        print('Finished evolving {0} generations.'.format(n_generations))
        print("Elapsed time was {0:.1f} seconds.".format(end_time - start_time))
        print(' ')
        print('Initial best distance: {0:.2f}'.format(initial_length))
        print('Final best distance:   {0:.2f}'.format(best_route.length))
        print('The best route went via:')
        op_file.write("Final Best Distance:"+str(best_route.length)+"\n")
        op_file.write("The best route via:"+str(best_route.pr_cits_in_rt())+" ")
        best_route.pr_cits_in_rt()
        op_file.close()
        

    def window_loop(self):
        # see http://stackoverflow.com/questions/459083/how-do-you-run-your-own-code-alongside-tkinters-event-loop
        self.window.after(0,self.GA_loop(self.n_generations, self.pop_size))
        self.window.mainloop()


    # Helper function for clearing terminal window
    def clear_term():
        os.system('cls' if os.name=='nt' else 'clear')


for city in list_of_cities:
    city.calculate_distances()




######## create and run an application instance:
app = App(n_generations=k_n_generations,pop_size=k_population_size)
