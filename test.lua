getinfo = io.popen('ls')
all = getinfo:read('*all')
 
local filenameList = io.open("filename.txt", "wb")
filenameList:write("Path/n")
filenameList:close()
 
filenameList = io.open("filename.txt", "a")
filenameList:write(all)
io.close(filenameList)
io.close(getinfo)
 
 
local filenameList = tab.Open([[../filename.txt]], "Path", false)
 
for index, Row in ipairs(filenameList) do
local pathlist = Row["Path"]
local  rowString = string.find(pathlist, ".txt")
if rowString ~= nil and rowString ~="" then
moon.CheckRowlength(pathlist, pathlist)
end
end
