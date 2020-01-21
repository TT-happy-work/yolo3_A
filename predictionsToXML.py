import os
import cv2
import xml.etree.ElementTree as ET

########################################################################################
# This code takes output of the network (directory of "predicted" txt files)
# and creates a directory with XML files for Casper.
# This is for helping the annotators, based on the network
########################################################################################

if __name__ == '__main__':
    predictedFolder = '/home/mayarap/Reccelite/ourRep/yolo3_A/mAP/predicted'  ## add here a folder with many predicted txt files (output of the network)
    outputFolder = './mAP/predictedOutput' ## add here a folder for xml files
    annotPath = "/home/mayarap/tamar_pc_DB/DBs/Reccelite_iter1/Taggings/Tagging1/" ## add here a folder with images

#    IMAGE_H = 2464
#    IMAGE_W = 3296
    subFolderList = os.listdir(predictedFolder)
    for singleImg in subFolderList:

        head, sep, tail = singleImg.partition('.')
        if tail == 'xml':
            continue
        imgName = head
        if not os.path.exists(annotPath + imgName):
            continue
        if os.path.exists(annotPath + imgName + '/' + imgName + '.jpg'):
            imgType = '.jpg'
        elif os.path.exists(annotPath + imgName + '/' + imgName + '.png'):
            imgType = '.png'
        else:
            imgType = '.' + input("Please enter type of image (i.e jpg) : ")

        fullImgName = annotPath + '/' + imgName + '/' + imgName + imgType  ## for Linux!! ##
        output_xml = outputFolder + '/' + imgName + '.xml'
        input_txt = predictedFolder + '/' + singleImg
        img = cv2.imread(fullImgName)  ## for Linux!! ##
        sizeX, sizeY, channels = img.shape

        if os.path.exists(output_xml):
            os.remove(output_xml)
        open(output_xml, 'a').close()

        ## for Linux!! ## location = annotPath + "/" + imgName
        location = r"\\rafael.local\cloud\Division\ibud_temuna_394\39484\Computer_Vision_For_Inteligent_Systems_ar1\RECCELITE\DB\Tagging1" + '\\' + imgName  ## for windows!! ##
        root = ET.Element('Project', Version="2.1.11.6")
        ProjectConfiguration = ET.SubElement(root, 'ProjectConfiguration')
        WorldPoints = ET.SubElement(root, 'WorldPoints')
        Appearances = ET.SubElement(root, 'Appearances')
        ProjectName = ET.SubElement(ProjectConfiguration, 'ProjectName').text = imgName
        ProjectLocation = ET.SubElement(ProjectConfiguration, 'ProjectLocation').text = location
        ProjectType = ET.SubElement(ProjectConfiguration, 'ProjectType').text = 'Tracker'
        ImageExtensions = ET.SubElement(ProjectConfiguration, 'ImageExtensions').text = 'Tiff|Tif|Jpeg|Jpg'
        MovieFolder = ET.SubElement(ProjectConfiguration, 'MovieFolder').text = location

        SkipValue = ET.SubElement(ProjectConfiguration, 'SkipValue').text = '1'
        NavFile = ET.SubElement(ProjectConfiguration, 'NavFile')
        ShowFramesPreview = ET.SubElement(ProjectConfiguration, 'ShowFramesPreview').text = 'true'
        NavDatum = ET.SubElement(ProjectConfiguration, 'NavDatum').text = 'Wgs84'
        NavUTMZone = ET.SubElement(ProjectConfiguration, 'NavUTMZone').text = '0'
        NavProjection = ET.SubElement(ProjectConfiguration, 'NavProjection').text = 'Utm'
        NavHeight = ET.SubElement(ProjectConfiguration, 'NavHeight').text = 'AboveSea'
        NavHemisphere = ET.SubElement(ProjectConfiguration, 'NavHemisphere').text = 'North'
        InterlaceType = ET.SubElement(ProjectConfiguration, 'InterlaceType').text = 'None'
        Properties = ET.SubElement(ProjectConfiguration, 'Properties')

        TargetProperties = ET.SubElement(Properties, 'TargetProperties')
        TargetProperty1 = ET.SubElement(TargetProperties, 'TargetProperty')
        Name1 = ET.SubElement(TargetProperty1, 'Name').text = 'Name'
        TargetType1 = ET.SubElement(TargetProperty1, 'TargetType').text = 'World'
        Type1 = ET.SubElement(TargetProperty1, 'Type').text = 'String'
        DefaultValue1 = ET.SubElement(TargetProperty1, 'DefaultValue')
        TargetProperty2 = ET.SubElement(TargetProperties, 'TargetProperty')
        Name2 = ET.SubElement(TargetProperty2, 'Name').text = 'Is Manual'
        TargetType2 = ET.SubElement(TargetProperty2, 'TargetType').text = 'Sensor'
        Type2 = ET.SubElement(TargetProperty2, 'Type').text = 'Boolean'
        DefaultValue2 = ET.SubElement(TargetProperty2, 'DefaultValue').text = 'true'
        TargetProperty3 = ET.SubElement(TargetProperties, 'TargetProperty')
        Name3 = ET.SubElement(TargetProperty3, 'Name').text = 'Grade'
        TargetType3 = ET.SubElement(TargetProperty3, 'TargetType').text = 'Sensor'
        Type3 = ET.SubElement(TargetProperty3, 'Type').text = 'Numeric'
        DefaultValue3 = ET.SubElement(TargetProperty3, 'DefaultValue').text = '100'
        TargetProperty4 = ET.SubElement(TargetProperties, 'TargetProperty')
        Name4 = ET.SubElement(TargetProperty4, 'Name').text = 'Is Dynamic'
        TargetType4 = ET.SubElement(TargetProperty4, 'TargetType').text = 'Sensor'
        Type4 = ET.SubElement(TargetProperty4, 'Type').text = 'Boolean'
        DefaultValue4 = ET.SubElement(TargetProperty4, 'DefaultValue').text = 'false'
        TargetProperty5 = ET.SubElement(TargetProperties, 'TargetProperty')
        Name5 = ET.SubElement(TargetProperty5, 'Name').text = 'Enter Hidden'
        TargetType5 = ET.SubElement(TargetProperty5, 'TargetType').text = 'Sensor'
        Type5 = ET.SubElement(TargetProperty5, 'Type').text = 'Boolean'
        DefaultValue5 = ET.SubElement(TargetProperty5, 'DefaultValue').text = 'false'
        TargetProperty6 = ET.SubElement(TargetProperties, 'TargetProperty')
        Name6 = ET.SubElement(TargetProperty6, 'Name').text = 'Exit Hidden'
        TargetType6 = ET.SubElement(TargetProperty6, 'TargetType').text = 'Sensor'
        Type6 = ET.SubElement(TargetProperty6, 'Type').text = 'Boolean'
        DefaultValue6 = ET.SubElement(TargetProperty6, 'DefaultValue').text = 'false'

        MarkedImage = ET.SubElement(Appearances, 'MarkedImage')
        ImgFileName = ET.SubElement(MarkedImage, 'ImgFileName').text = location + '\\' + imgName + imgType
        ImgSubFolderPath = ET.SubElement(MarkedImage, 'ImgSubFolderPath')
        ImageSize = ET.SubElement(MarkedImage, 'ImageSize')
        X = ET.SubElement(ImageSize, 'X').text = str(sizeX)
        Y = ET.SubElement(ImageSize, 'Y').text = str(sizeY)
        SensorPointWorldPointPairs = ET.SubElement(MarkedImage, 'SensorPointWorldPointPairs')

        tree = ET.ElementTree(root)
#        ET.dump(tree)

        counter = 0
        with open(input_txt, 'r') as f:
            txt = f.readlines()
            for line in txt:
                line_to_list = line.split()
                objClass = line_to_list[0]
                x1 = line_to_list[1]
                y1 = line_to_list[2]
                x3 = line_to_list[3]
                y3 = line_to_list[4]
                x2 = x3
                y2 = y1
                x4 = x1
                y4 = y3

                WorldPoint = ET.SubElement(WorldPoints, 'WorldPoint')
                ID = ET.SubElement(WorldPoint, 'ID').text = str(counter)
                Name = ET.SubElement(WorldPoint, 'Name').text = objClass

                SensorPointWorldPointPair = ET.SubElement(SensorPointWorldPointPairs, 'SensorPointWorldPointPair')
                First = ET.SubElement(SensorPointWorldPointPair, 'First')
                Shape = ET.SubElement(First, 'Shape').text = 'Polygon'
                Coordinate = ET.SubElement(First, 'Coordinate')
                X = ET.SubElement(Coordinate, 'X').text = str(x1)
                Y = ET.SubElement(Coordinate, 'Y').text = str(y1)
                Coordinate = ET.SubElement(First, 'Coordinate')
                X = ET.SubElement(Coordinate, 'X').text = str(x2)
                Y = ET.SubElement(Coordinate, 'Y').text = str(y2)
                Coordinate = ET.SubElement(First, 'Coordinate')
                X = ET.SubElement(Coordinate, 'X').text = str(x3)
                Y = ET.SubElement(Coordinate, 'Y').text = str(y3)
                Coordinate = ET.SubElement(First, 'Coordinate')
                X = ET.SubElement(Coordinate, 'X').text = str(x4)
                Y = ET.SubElement(Coordinate, 'Y').text = str(y4)
                IsManual = ET.SubElement(First, 'IsManual').text = 'True'
                Grade = ET.SubElement(First, 'Grade').text = '100'
                IsDynamic = ET.SubElement(First, 'IsDynamic').text = 'False'
                EnterHidden = ET.SubElement(First, 'EnterHidden').text = 'False'
                ExitHidden = ET.SubElement(First, 'ExitHidden').text = 'False'

                Second = ET.SubElement(SensorPointWorldPointPair, 'Second')
                WorldPointId = ET.SubElement(Second, 'WorldPointId').text = str(counter)
                OrdinalNumber = ET.SubElement(Second, 'OrdinalNumber').text = str(0)

                counter = counter + 1
        ET.dump(tree)

        tree.write(output_xml)

