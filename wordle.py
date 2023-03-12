import os
import time
import string
import pandas as pd
from tqdm import tqdm



class WordleMe():

    def __init__(self):

        self.value_correct = '🟩'
        self.value_wrong_position = '🟨'
        self.value_incorrect = '⬛️'


        with open('solutions_20220215.txt', 'r') as f:
            block_of_text = f.read()
            no_quotes = block_of_text.replace('"','')
            self.solutions_sequence = no_quotes.split(',')
            self.solutions = sorted(self.solutions_sequence)
            self.solutions = [w.upper() for w in self.solutions]
            # print(f'Number of solutions: {len(self.solutions)}')

        with open('herrings_20220215.txt', 'r') as f:
            block_of_text = f.read()
            no_quotes = block_of_text.replace('"','')
            self.herrings = no_quotes.split(',')
            self.herrings = [w.upper() for w in self.herrings]
            # print(f'Number of herrings: {len(self.herrings)}')



    def save_word_beginnings(self):

        beginnings = []

        for word in self.solutions:
            beginnings += [word[:2]]

        dfb = pd.DataFrame(beginnings, columns=['beginnings'])
        counts_beginnings = pd.DataFrame(dfb.groupby('beginnings').size(), columns=['N'])
        counts_beginnings['percentage'] = counts_beginnings['N'] / counts_beginnings['N'].sum()
        counts_beginnings['first_letter'] = counts_beginnings.index.str[0]
        counts_beginnings['second_letter'] = counts_beginnings.index.str[1]

        print(f'Possible beginnings: {len(counts_beginnings)}')
        print(counts_beginnings.sort_values(by='N', ascending=False).head(20))
        counts_beginnings.sort_values(by='N', ascending=False).to_clipboard()

        print('')


    def save_word_endings(self):
        endings = []

        for word in self.solutions:
            endings += [word[-2:]]

        dfe = pd.DataFrame(endings, columns=['endings'])
        counts_endings = pd.DataFrame(dfe.groupby('endings').size(), columns=['N'])
        counts_endings['percentage'] = counts_endings['N'] / counts_endings['N'].sum()
        counts_endings['fourth_letter'] = counts_endings.index.str[0]
        counts_endings['fifth_letter'] = counts_endings.index.str[1]

        print(f'Possible endings: {len(counts_endings)}')
        print(counts_endings.sort_values(by='N', ascending=False).head(20))
        # counts_endings.sort_values(by='N', ascending=False).to_clipboard()


    def save_sorted_list(self):
        """
        Write the solution list in alphabetical order to a new file
        """

        with open('solutions_20220215_sorted.txt', 'w') as f:
            f.write('\n'.join(sorted(self.solutions)))

        with open('herrings_20220215_sorted.txt', 'w') as f:
            f.write('\n'.join(sorted(self.herrings)))

    def frequency(self):

        df = pd.DataFrame(columns=['words_containing_letter'], index=list(string.ascii_uppercase))

        for c in string.ascii_uppercase:

            letter_in_word = 0
            for w in self.solutions:
                if c in w:
                    letter_in_word += 1

            df.loc[c, 'words_containing_letter'] = letter_in_word


        df['perc'] = df['words_containing_letter'] / len(self.solutions)

        df = df.sort_values(by='perc', ascending=False)
        # print(df)
        # df.to_csv('most_common_letters.csv')

        return df


    def solution_with_common_letters(self):
        """
        Solutions that contain 'orate', the most common letters
        """

        for idx, word in enumerate(self.solutions):
            # print(''.join(sorted(word)))

            if ''.join(sorted(word)) == 'aeors':
                print(word)



    def solution_stats(self):
        """
        Solutions that contain a lot of common letters
        """

        freq = self.frequency()
        scores = pd.DataFrame(
            index=self.solutions
            , columns=[
                'letter_freq_score'
                , 'num_unique'
                , 'num_vowels'
                , 'letter_0'
                , 'letter_1'
                , 'letter_2'
                , 'letter_3'
                , 'letter_4'
                ]
            )

        scores.index.name = 'solution'

        for idx, word in enumerate(self.solutions):
            word_score = 0
            unique_letters = []
            vowels = []

            for letter in word:
                word_score += freq.loc[letter, 'perc']

                if letter not in unique_letters:
                    unique_letters += [letter]

                if letter in ['A', 'E', 'I', 'O', 'U']:
                    vowels += [letter]

            # print(word)
            # print(unique_letters)
            # print()

            scores.loc[word, 'letter_freq_score'] = word_score
            scores.loc[word, 'num_unique'] = len(unique_letters)
            scores.loc[word, 'num_vowels'] = len(vowels)

            for position, letter in enumerate(word):
                scores.loc[word, f'letter_{position}'] = letter

            for character in string.ascii_uppercase:
                scores.loc[word, f'has_{character}'] = character in word

            for character in string.ascii_uppercase:
                scores.loc[word, f'count_{character}'] = self.count_letter_occurances_in_word(word, character)

        # print(scores[scores['num_unique'] == 5].sort_values(by='letter_freq_score', ascending=False).head(20))
        # print(scores[(scores['num_unique'] == 5) & (scores['num_vowels'] == 1)].sort_values(by='letter_freq_score', ascending=False).head(20))

        scores['difficulty_percentile'] = scores['letter_freq_score'].rank(pct=True, ascending=False)
        # print(scores.sort_values(by='letter_freq_score'))
        # print(scores.loc['ultra'])
        # scores.to_clipboard()

        unique_stats = pd.DataFrame(scores.groupby('num_unique').size(), columns=['N'])
        unique_stats['perc'] = unique_stats['N'] / unique_stats['N'].sum()
        print(unique_stats)

        scores.to_csv('scores.csv')



    def count_letter_occurances_in_word(self, word, letter):
        """Return the number of times that 'letter' appears in the 'word' """

        count_occurances = 0

        for j in range(5):
            if word[j] == letter:
                count_occurances += 1

        return count_occurances



    def difficulty_by_day(self):
        """
        How difficult is every word?

        ulcer = 2022-02-11
        """

        word_rank = pd.DataFrame(index=self.solutions, columns=['letter_freq_score', 'rank', 'percentile'])
        
        for word in enumerate(self.solutions):

            word_rank.loc[word, 'letter_freq_score'] = 1



    def corpus_frequency(self):

        word_frequency = pd.read_excel('wordFrequency.xlsx', sheet_name='3 wordForms')

        in_word_list = 0
        for word in self.solutions:
            if word in word_frequency['word'].tolist():
                in_word_list += 1

        print(in_word_list / len(self.solutions))



    def previous_in_sequence(self):
        """
        Is the sequence still the same? Check without spoiling an answer by revealing the answer from yesterday
        """

        todays_solution = 'aroma'
        yesterdays_solution = self.solutions_sequence[self.solutions_sequence.index(todays_solution) - 1]
        print(f'yesterdays_solution: {yesterdays_solution}')



    def compare_two_lists(self, list_left, list_right):
        
        in_left_but_not_right = []

        for word in list_left:
            if word not in list_right:
                in_left_but_not_right += [word]

        print(f'in_left_but_not_right: {in_left_but_not_right} ({len(in_left_but_not_right)} words)')


        in_right_but_not_left = []

        for word in list_right:
            if word not in list_left:
                in_right_but_not_left += [word]

        print(f'in_right_but_not_left: {in_right_but_not_left} ({len(in_right_but_not_left)} words)')



    def compare_lists_to_current(self, comparison_date='20220215'):
        """
        Compare a set of Wordle lists to 
        """

        with open(f'solutions_{comparison_date}.txt', 'r') as f:
            block_of_text = f.read()
            no_quotes = block_of_text.replace('"','')
            solutions_sequence = no_quotes.split(',')
            solutions_sequence = [w.upper() for w in solutions_sequence]
            solutions_comparison = sorted(solutions_sequence)

        with open(f'herrings_{comparison_date}.txt', 'r') as f:
            block_of_text = f.read()
            no_quotes = block_of_text.replace('"','')
            herrings_comparison = no_quotes.split(',')
            herrings_comparison = [w.upper() for w in herrings_comparison]

        print('\nSolution changes:')
        self.compare_two_lists(solutions_comparison, self.solutions)
        print('\nHerring changes:')
        self.compare_two_lists(herrings_comparison, self.herrings)
        print()



    def guess_analysis(self, guesses, solution=None, debug=False, terminal_format=True):
        """
        For a list of guesses, show how well each guess performed at limiting the field of possible solutions.

        solution= the word that is the solution to the puzzle. If not given, assume the last guess is the solution
        """

        print('\n\n\n~~~~~ ~~~~~ Wordle Guess Analysis ~~~~~ ~~~~~')

        if not solution:
            solution = guesses[-1]

        solution = solution.upper()
        guesses = [g.upper() for g in guesses]

        psdf = pd.read_csv('scores.csv')
        psdf['possible'] = 1
    
        for guess_num, g in enumerate(guesses):
            print(f'\nGuess {guess_num+1}: {g}')

            response = self.process_guess(solution, g)
            print('         ', end='')
            self.print_emoji(response)
            self.is_guess_playable(g)

            # self.is_guess_on_list(g)

            psdf, _ = self.eliminate_solutions(psdf, solution, g, verbose_es=True, debug=debug, terminal_format=terminal_format)



    def process_guess(self, solution, guess):
        """
        Guess at a solution. Return green, yellow, or gray
        """

        response = [self.value_incorrect] * 5

        unmatched_letters_in_solution = []

        for position, letter in enumerate(guess):
            if solution[position] == letter:
                response[position] = self.value_correct
            else:
                unmatched_letters_in_solution += [solution[position]]

        for position, letter in enumerate(guess):
            if response[position] == self.value_correct:
                continue
            elif letter in unmatched_letters_in_solution:
                response[position] = self.value_wrong_position
        
        return response



    def is_guess_playable(self, guess):
        """
        Raise an exception if the guess is not on the lists of playable words
        """

        if (guess not in self.solutions) and (guess not in self.herrings):
            raise Exception(f'"{guess}" is not a playable word.')



    def is_guess_on_list(self, guess):
        """
        Print information about which list the guess is on.
        """

        if (guess not in self.solutions) and (guess in self.herrings):
            print(f'"{guess}" is a playable word but not on the list of solutions, so it\'s not an optimal guess.')

        elif guess in self.solutions:
            print(f'"{guess}" is on the list of solutions.')



    def print_emoji(self, emoji_sequence):
        """
        Print the response emoji sequence to the terminal.
        """

        for e in emoji_sequence:
            print(e, end='')

        print()



    def print_word_list(self, word_list, solution=None, max_chars_per_line=80, highlight_solution=False):
        """
        Print a list of words in a pretty block
        """

        chars_used = 0

        for idx, word in enumerate(word_list):

            if highlight_solution:

                if word == solution:
                    print('>>>' + word + '<<<', end=' ')
                    chars_used += len(word) + 6
                else:
                    print(word, end=' ')
                    chars_used += len(word) + 1


            print(word, end=' ')
            chars_used += len(word) + 1

            if chars_used >= max_chars_per_line:
                print()
                chars_used = 0


        print()



    def eliminate_solutions(self, psdf, solution, guess, verbose_es=False, debug=False, terminal_format=True):
        """
        Show how many possible solutions this guess has eliminated
        """

        before_guess_list = psdf[psdf['possible'] == 1]['solution'].tolist()
        before_guess_count = len(before_guess_list)

        if verbose_es:
            if guess != solution:
                if guess in before_guess_list:
                    print(f'"{guess}" is on the list of possible solutions at the start of this round, well played!')
                else:
                    print(f'"{guess}" is not on the list of possible solutions at the start of this round, so it\'s not an optimal guess.')

                if guess in self.herrings:
                    print(f'Additionally, "{guess}" will never be a solution, though it is a playable word.')

        response = self.process_guess(solution, guess)

        for position, status in enumerate(response):

            letter = guess[position]

            if status == self.value_correct:
                psdf.loc[psdf[f'letter_{position}'] != letter, 'possible'] = 0

                if debug:
                    print(f'position {position}, guess "{letter}", value_correct')
                    self.print_word_list(psdf[psdf['possible'] == 1]['solution'].tolist())

            elif status == self.value_wrong_position:
                psdf.loc[(psdf[f'has_{letter}'] == False) | (psdf[f'letter_{position}'] == letter), 'possible'] = 0

                if debug:
                    print(f'position {position}, guess "{letter}", value_wrong_position')
                    self.print_word_list(psdf[psdf['possible'] == 1]['solution'].tolist())

            elif status == self.value_incorrect:
                # This letter cannot be in the guessed position
                psdf.loc[psdf[f'letter_{position}'] == letter, 'possible'] = 0

                # Also loop through each position in the response. 
                # If the value isn't green, then this incorrect letter cannot be in that position. 
                # If the value is green, then it is possible for this incorrect letter to be in that position.

                for j in range(5):
                    if response[j] != self.value_correct:
                        psdf.loc[psdf[f'letter_{j}'] == letter, 'possible'] = 0

                if debug:
                    print(f'position {position}, guess "{letter}", value_incorrect')
                    self.print_word_list(psdf[psdf['possible'] == 1]['solution'].tolist())

        after_guess_count = psdf['possible'].sum()

        if after_guess_count == 0:
            raise Exception('No possible solutions remain. Something went wrong in the word elimination logic.')

        eliminated = before_guess_count - after_guess_count
        elim_perc = eliminated / before_guess_count

        if verbose_es:

            # If the guess is the solution AND the only possibility at this guess, skip this message
            if not (guess == solution and before_guess_count == 1):
                plural = 's' if before_guess_count > 1 else ''
                print(f'Guess "{guess}" eliminated {eliminated} of {before_guess_count} solution{plural} ({elim_perc:.1%})')

            if guess == solution:
                print(f'"{guess}" is correct!')
            else:
                noun = 's' if after_guess_count > 1 else ''
                verb = '' if after_guess_count > 1 else 's'
                print(f'{after_guess_count} solution{noun} remain{verb}: ')

                if terminal_format:
                    max_chars_per_line=80
                else:
                    max_chars_per_line=10000

                self.print_word_list(psdf[psdf['possible'] == 1]['solution'].tolist(), solution, max_chars_per_line=max_chars_per_line)

        return psdf, elim_perc



    def analyze_opening_guesses(self):
        """
        Cycle through all opening guesses to see which consistently eliminate the most solutions

        Save output to CSV on each cycle for safety because this function takes like 8 hours to complete

        Each row is one possible solution
        Each column is a possible opening
        """

        # Create the output DataFrame if it doesn't already exist
        output_filename = 'opening_df.csv'

        if os.path.exists(output_filename):
            opening_df = pd.read_csv(output_filename, index_col='solution')
        else:
            opening_df = pd.DataFrame(index=self.solutions, columns=self.solutions)
            opening_df.index.name = 'solution'
            opening_df.columns.name = 'opening'


        psdf_static = pd.read_csv('scores.csv')
        psdf_static['possible'] = 1

        # Find solutions in the DataFrame that have not yet had openings tested against them
        words_tested = opening_df[self.solutions].notnull().sum(axis=1)
        words_remaining = words_tested[words_tested == 0].index.tolist()

        for idx, solution in enumerate(words_remaining):

            print(f'Testing best opening word for: {solution}')

            opening_elim_row = self.one_guess_all_solutions(solution, verbose_ogas=False, subset_only=False).transpose().values
            opening_df.loc[solution, :] = opening_elim_row[0]

            opening_df.to_csv(output_filename)
            # opening_df.to_csv(output_filename, index_label='solution')

            # if idx % 50 == 0:
            #     print('Sleeping for 3 minutes...')
            #     time.sleep(180)

            if idx > 3:
                break

        print('Opening guess dataset complete.')



    def one_guess_all_solutions(self, guess, verbose_ogas=False, subset_only=False):
        """
        Run one opening guess through all the possible solutions as an opening word
        """

        psdf = pd.read_csv('scores.csv')
        psdf['possible'] = 1
        guess_elim_perc = pd.DataFrame(index=self.solutions, columns=[guess])

        if verbose_ogas:
            iterator = tqdm(self.solutions)
        else:
            iterator = self.solutions

        for idx, solution in enumerate(iterator):

            if subset_only and idx > 10:
                break

            # Ignore situations where the guess is perfect
            if guess == solution:
                continue

            _, elim_perc = wm.eliminate_solutions(psdf.copy(), solution, guess, verbose_es=False)
            guess_elim_perc.loc[solution, guess] = elim_perc

        if verbose_ogas:        
            print(f'On average, guess "{guess}" eliminates {guess_elim_perc[guess].mean():.1%} of solutions when played as the first guess.')

        # todo: make a histogram

        return guess_elim_perc



    def best_opening_word(self):

        df = pd.read_csv('opening_df_complete.csv', index_col='solution')

        openings_median = df[self.solutions].median(numeric_only=True, axis=0)
        openings_median.name = 'elim_perc_median'
        openings_mean = df[self.solutions].mean(numeric_only=True, axis=0)
        openings_mean.name = 'elim_perc_mean'

        openings_stats = pd.merge(openings_median, openings_mean, how='inner', left_index=True, right_index=True)
        openings_stats['elim_perc_median_rank'] = openings_stats['elim_perc_median'].rank(ascending=False)
        openings_stats['elim_perc_mean_rank'] = openings_stats['elim_perc_mean'].rank(ascending=False)
        
        openings_stats['elim_more_than_50'] = (df[self.solutions] > 0.5).sum(axis=0) / len(df)
        openings_stats['elim_more_than_50_rank'] = openings_stats['elim_more_than_50'].rank(ascending=False)
        
        openings_stats.index.name = 'guess'
        output_columns = [
            'elim_perc_mean'
            , 'elim_perc_mean_rank'
            , 'elim_perc_median'
            , 'elim_perc_median_rank'
            , 'elim_more_than_50'
            , 'elim_more_than_50_rank'
        ]
        openings_stats[output_columns].sort_values(by=['elim_more_than_50_rank', 'elim_perc_mean_rank']).to_csv('openings_stats.csv')
        print('Stats saved to: openings_stats.csv')

        # print(sum(df['RAISE'] < 0.6))



    def share_four_letters(self):
        """
        Identify groups of solutions that share 4 letters with each other

        This is working well for first and last. But how to handle clusters like SMELT and SPELT?
        """

        psdf = pd.read_csv('scores.csv', index_col='solution')
        psdf['match_first_four'] = None
        psdf['match_first_four_count'] = None
        psdf['match_last_four'] = None
        psdf['match_last_four_count'] = None

        for solution in tqdm(self.solutions):
            match_first_four = [solution]
            match_last_four = [solution]
            
            for comparison in self.solutions:

                if comparison == solution:
                    continue

                if solution[:4] == comparison[:4]:
                    match_first_four += [comparison]

                if solution[-4:] == comparison[-4:]:
                    match_last_four += [comparison]

            psdf.loc[solution, 'match_first_four'] = ', '.join(sorted(match_first_four))
            psdf.loc[solution, 'match_first_four_count'] = len(match_first_four)
            psdf.loc[solution, 'match_last_four'] = ', '.join(sorted(match_last_four))
            psdf.loc[solution, 'match_last_four_count'] = len(match_last_four)

        psdf[['match_first_four', 'match_first_four_count', 'match_last_four', 'match_last_four_count']].to_csv('share_four_letters.csv')



    def cluster_four_letters(self):

        psdf = pd.read_csv('scores.csv', index_col='solution')

        cluster_long = pd.DataFrame(columns=['cluster', 'word'])

        for solution in tqdm(self.solutions):

            for comparison in self.solutions:

                if comparison == solution:
                    continue

                letters_shared = 0
                shared_list = ['_'] * 5

                for i in range(5):
                    if solution[i] == comparison[i]:
                        letters_shared += 1
                        shared_list[i] = solution[i]

                if letters_shared == 4:

                    shared_word = ''.join(shared_list)
                    new_row = {'cluster': shared_word, 'word': comparison}
                    cluster_long = cluster_long.append(new_row, ignore_index=True)

            # if len(cluster_long) > 20:
            #     break

        cluster_long = cluster_long.sort_values(by='word').drop_duplicates()
        cluster_wide = cluster_long.groupby('cluster').agg(
            num_words=('cluster', 'size')
            , solution_list=('word', ', '.join)
            )
        print(cluster_wide.sort_values(by='num_words', ascending=False))
        # cluster_wide.to_csv('share_four_letters.csv')





if __name__ == '__main__':

    wm = WordleMe()

    # wm.solution_with_common_letters()
    # wm.frequency()

    # wm.solution_stats()
    
    # wm.corpus_frequency()

    # wm.previous_in_sequence()

    # wm.compare_lists_to_current()

    # wm.save_sorted_list()

    # wm.guess_analysis(['arise', 'white', 'glide', 'olive']) # Devin 2022-04-23



    # wm.analyze_opening_guesses()
    wm.best_opening_word()


    # wm.one_guess_all_solutions('RAISE', verbose_ogas=True)
    # wm.one_guess_all_solutions('NOISE', verbose_ogas=True)
    # wm.one_guess_all_solutions('CABLE', verbose_ogas=True)

    # wm.share_four_letters()
    # wm.cluster_four_letters()

    

    # todo: histogram of positions of letters
    # todo: maybe a difficulty score of each solution based on the percentage of solutions eliminated
    #       from each opening guess. like, opposite axis mean of the best opening word


