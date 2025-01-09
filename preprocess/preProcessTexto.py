import re

pittPath = '/Pitt/'

def preProcessFile(filePath):

  file = open(filePath, 'r')
  lines = file.readlines()


  traditional = False
  utt = ''


  for line in lines:
      if line[0:4] == '*PAR':
          if not '[+ exc]' in line:
              traditional = True
              utt += line[6:-1]
      elif traditional and line[0:4] == '%mor':
          end = -2
          while utt[end].isnumeric():
              end -= 1
          end -= 1
          while utt[end].isnumeric():
              end -= 1
          traditional = False
          utt = utt[:end]
      elif traditional:
          if not '[+ exc]' in line:
              utt += line[:-1]

  repetitions = utt.count('[/]')
  retraicings = utt.count('[//]')
  spause = utt.count('(.)')
  mpause = utt.count('(..)')
  lpause = utt.count('(...)')
  tpauses = spause + mpause + lpause
  unintelligible = utt.count('xxx')
  gramerros = utt.count('[+ gram]')
  doubts = utt.count('&uh') + utt.count('&um')

  utt = utt.replace('\n', '')
  utt = utt.replace('\t', ' ')
  utt = utt.replace('(', '')
  utt = utt.replace(')', '')
  utt = utt.replace('+<', '')
  utt = utt.replace('&', '')

  utt = re.sub('[\<\[].*?[\>\]]', '', utt)
  return [utt, repetitions, retraicings, spause, mpause, lpause, tpauses, unintelligible, gramerros, doubts]


def preprocessData():
  categories = ['controlCha.txt', 'dementiaCha.txt']
  for cat in categories:
    path = pittPath + cat

    index = open(path, 'r')
    files = index.readlines()
    for file in files:
      properties = preProcessFile(file[:-1])
      out = open(file[:-4] + 'txt', 'w')
      for i, val in enumerate(properties):
          if i != 8 and i != 9:
              out.write(str(val) + '\n')
      out.close()

if __name__ == "__main__":
    preprocessData()
