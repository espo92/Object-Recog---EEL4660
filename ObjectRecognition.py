# import necessary libraries
import numpy
import matplotlib.pyplot as pyplot
import cv2
import glob
from operator import itemgetter

TEMPLATE_MATCHING_METHOD = cv2.TM_CCOEFF_NORMED

DATABASE_LOC = "Database\\"
QUERY_IMG_FILE_NAME = "ukbench00000.jpg"
TEST_FILE_PATERN = "ukbench*.jpg"

CORECT_MATCHES = [QUERY_IMG_FILE_NAME, "ukbench00001.jpg", "ukbench00002.jpg", "ukbench00003.jpg"]

def main():
    print "Loading files from: " + DATABASE_LOC
    print "Query image name: " + QUERY_IMG_FILE_NAME
    print "File pattern: " + TEST_FILE_PATERN

    #load query image
    queryImg = cv2.imread(DATABASE_LOC + QUERY_IMG_FILE_NAME)
    
    #Get a list of all files names that follow the patern of our images
    strFileNames = glob.glob(DATABASE_LOC + TEST_FILE_PATERN)

    #read in all files from the database and store them in imageList
    imageList = []
    for curPath in strFileNames:
        curImage = cv2.imread(curPath)
        imageList.append((curPath.strip(DATABASE_LOC), curImage))

    print "Done loading images"

    #preform template matching on each image
    print "Running Template Matching comparison"
    tempMatchResults = []
    tempMatchResults = CompareTemplateMatching(queryImg, imageList)

    #preform color histogram comparison on each image
    print "Running Color Histogram comparison"
    colorHistResults = []
    colorHistResults = CompareColorHist(queryImg, imageList)

    #perform SIFT algorithem on each image and save good results
    print "Running SIFT algorithm"
    siftResults = []
    siftResults = CompareSIFT(queryImg, imageList)

    print "Finished image comparisons"

    #prints out the scores for each method
    ScoreResults(tempMatchResults, colorHistResults, siftResults)

    # Wait for the user to hit enter then close everything
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

def CompareTemplateMatching(queryImg, imageList):
    results = []

    for name, curImage in imageList:
        #Perform template matching: Slide template image over scene image and
        #get scores for matches at each position
        result = cv2.matchTemplate(curImage, queryImg, TEMPLATE_MATCHING_METHOD) 

        #Find best scores
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        matchValue = maxVal
        if TEMPLATE_MATCHING_METHOD in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            matchValue = 1.0 - minVal

        #Save off the name, match value and current image
        #multiply maxVal by 100 to convert the match from 0% to 100%
        results.append((name, max(matchValue*100.0, 0.0), curImage))

    return results

def CompareColorHist(queryImg, imageList):

    #convert the query image to HSV and store it for use later
    queryImgHsv = cv2.cvtColor(queryImg,  cv2.cv.CV_BGR2HSV)
    queryImgHist = cv2.calcHist([queryImgHsv],[0],None,[256],[0,255])

    #loop through each element of the imageList and compare against the queryImg
    #Store the good images in colorHistResults
    results = []
    for name, curImage in imageList:
        #convert the current image to HSV and calculate its color histogram
        curImageHsv = cv2.cvtColor(curImage,  cv2.cv.CV_BGR2HSV)
        curImageHist = cv2.calcHist([curImageHsv],[0],None,[256],[0,255])

        #compare the histograms of the query image and of the curent image
        result = cv2.compareHist(queryImgHist, curImageHist, cv2.cv.CV_COMP_CORREL)
        
        #Save off the name, match value and current image
        #multiply maxVal by 100 to convert the match from 0% to 100%
        results.append((name, result*100.0, curImage))

    #return the good results
    return results

def CompareSIFT(queryImg, imageList):
    # Initiate SIFT detector
    sift = cv2.SIFT()

    #find the keypoints and descriptors of the query image once
    queryImgKp, queryImgDes = sift.detectAndCompute(queryImg,None)
    
    #loop through each element of the imageList and compare against the queryImg
    #Store the good images in siftResult
    results = []
    for name, curImage in imageList:
        #build dictionarys containing the algorithem to use
        #specify the algorithm as 0 which is FLANN_INDEX_KDTREE
        indexParams = dict(algorithm = 0, trees = 5)
        searchParams = dict(checks = 50)

        #find the keypoints and descriptors of the current image
        curImageKp, curImageDes = sift.detectAndCompute(curImage,None)

        #generate the flann matcher based upon the specified algorithm and search params
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)

        #perform flann matching between the query image and the current image
        matches = flann.knnMatch(queryImgDes,curImageDes,k=2)

        #use lowes cost ratio testing to determin if the matched point is valid
        #if the matched keypoint is valid then add it to the goodKP array
        goodKp = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                goodKp.append(m)
        
        #Save off the name, match value and current image
        #convert the number of good keypoints to a percent.
        #4265 is the maximum number of keypoints that can be returned.
        #If we receive more than 10 matching key points it is very likely that this is a match.
        #to account for this we will assum that anything over 100 matches is at least a 90% match
        numMatches = len(goodKp)
        if numMatches > 100:
            percent = (numMatches * 10.0 / 4265.0) + 90.0
            results.append((name, percent, curImage))
        else:
            percent = numMatches / 100.0 * 90.0
            results.append((name, percent, curImage))

    #return the good results
    return results

def ScoreResults(tempMatchResults, colorHistResults, siftResults):
    
    #plot each methods results with matplotlib
    pyplot.plot([x[1] for x in tempMatchResults])
    pyplot.plot([x[1] for x in colorHistResults])
    pyplot.plot([x[1] for x in siftResults])
    pyplot.legend(["Template Matching", "Color Histogram", "SIFT"])
    pyplot.ylabel("Percent Match")
    pyplot.xlabel("Image Number")
    pyplot.title("Matching Results")

    #print out the percentages for each image in order
    print "\nScoring Results:"
    print "Image Name           Template Matching    Color Histogram    SIFT"
    for x in range(len(tempMatchResults)):
        print "%-20s %-20s %-18s %-16s" % (tempMatchResults[x][0], tempMatchResults[x][1], colorHistResults[x][1], siftResults[x][1])

    #Compute the average score accross all algorithms
    avgList = []
    for x in range(len(tempMatchResults)):
        average = (tempMatchResults[x][1] + colorHistResults[x][1] + siftResults[x][1])/3.0
        avgList.append((tempMatchResults[x][0], average))
        
    #sort the arrays in decending order
    tempMatchResults = sorted(tempMatchResults, key=itemgetter(1), reverse=True)
    colorHistResults = sorted(colorHistResults, key=itemgetter(1), reverse=True)
    siftResults = sorted(siftResults, key=itemgetter(1), reverse=True)

    print "\nHighest Ranked Images:"
    print "Template Matching    Color Histogram      SIFT"
    for x in range(4):
        print "%-20s %-20s %-20s" % (tempMatchResults[x][0], colorHistResults[x][0], siftResults[x][0])

    print "\nFinal Scores Individual:"

    #Compare against the threshold to see if this is a valid match
    tempScore = 0
    for x in range(4):
        name, score, img = tempMatchResults[x]
        if name in CORECT_MATCHES:
            tempScore = tempScore + 1
    print "Template Matching: " + str(tempScore)

    #If the highest 4 results are in the correct matches list then increment the score and print it
    tempScore = 0
    for x in range(4):
        name, score, img = colorHistResults[x]
        if name in CORECT_MATCHES:
            tempScore = tempScore + 1
    print "Color Histogram: " + str(tempScore)

    #If the highest 4 results are in the correct matches list then increment the score and print it
    tempScore = 0
    for x in range(4):
        name, score, img = siftResults[x]
        if name in CORECT_MATCHES:
            tempScore = tempScore + 1
    print "SIFT: " + str(tempScore)
    
    #If the highest 4 results are in the correct matches list then increment the score and print it
    print "\nFinal overall scores:"
    tempScore = 0
    for x in range(4):
        name, score = avgList[x]
        print "%-20s %-20s" % (name, score)
        if name in CORECT_MATCHES:
            tempScore = tempScore + 1
    print "Overall: " + str(tempScore)

    #display the plot
    pyplot.show()

if __name__ == "__main__":
    main()