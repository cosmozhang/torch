
local function genNumber(s)

	local val = tonumber(s)
	if val == nil then
		return 0
	else
		return val
	end
end

local function genString(s)
	return s
end

local function genTable(s)
	return assert(loadstring("return { " .. s .. " }"))()
end

local function genFunction0(s)
	return assert(loadstring("return function() return " .. s .. " end"))()
end

local function genFunction1(s)
	return assert(loadstring("return function(z) return " .. s .. " end"))()
end

local function genFunction2(s)
	return assert(loadstring("return function(a,b) return " .. s .. " end"))()
end

local function genFunction3(s)
	return assert(loadstring("return function(a,b,c) return " .. s .. " end"))()
end

local function genBoolean(s)
	local val = tonumber(s)
	if val == nil or val == 0 then
		return false
	else
		return true
	end
end

local function genValue(s)
	return assert(loadstring("return " .. s))()
end

local conv_map =
{
	n=genNumber,
	s=genString,
	t=genTable,
	v=genValue,
	f0=genFunction0,
	f1=genFunction1,
	f2=genFunction2,
	f3=genFunction3,
	c=nil, --ignore the column
	b=genBoolean,
	kn=genNumber,
	ks=genString,
}

local function getFieldInfo(field)
	local pos = string.find(field, '_')
	local isKey = false
	local name = nil
	local datatype = nil
	if pos then
		name = string.sub(field, 1, pos - 1)
		datatype = string.sub(field, pos + 1)
		if string.sub(datatype, -1) == '\n' then
			datatype = string.sub(datatype, 1, #datatype - 1)
		end
		if string.sub(datatype, -1) == '\r' then
			datatype = string.sub(datatype, 1, #datatype - 1)
		end
		if datatype == "kn"	or datatype == "ks" then
			isKey = true
		end
		if string.byte(name) == string.byte("$")
			or string.byte(name) == string.byte("#") then
			name = string.sub(name, 2)
		end
		return name, conv_map[datatype], isKey
	else
		name = field
		if string.sub(name, -1) == '\n' then
			name = string.sub(name, 1, #name - 1)
		end
		if string.sub(name, -1) == '\r' then
			name = string.sub(name, 1, #name - 1)
		end
		if string.byte(name) == string.byte("$")
			or string.byte(name) == string.byte("#") then
			name = string.sub(name, 2)
		end
		return name, conv_map["n"], isKey
	end
end

local function getFieldLocation(field)
	local location = 2
	if string.byte(field) ==  string.byte("$") then
		location = 1
	elseif string.byte(field) ==  string.byte("#") then
		location = 3
	else
		location = 2
	end

	return location
end


local function fromCSV (s, head)
	s = s .. ',' -- ending comma
	local t = {} -- table to collect fields
	local fieldstart = 1
	local index = 1
	local val = nil
	local name = nil
	local conv_func = nil
	local key = nil
	local key1 = nil
	local isKey = false

	repeat
		-- next field is quoted? (start with `"'?)
		if string.find(s, '^"', fieldstart) then
			local a, c
			local i = fieldstart
			repeat
				-- find closing quote
				a, i, c = string.find(s, '"("?)', i+1)
			until c ~= '"' -- quote not followed by quote?

			if not i then error('unmatched "') end
			local f = string.sub(s, fieldstart+1, i-1)
			val = string.gsub(f, '""', '"')
			if head then
				name, conv_func, isKey = getFieldInfo(head[index])
				if conv_func then
					if isKey then
						if key then
							key1 = conv_func(val)
							t[name] = key1
						else
							key = conv_func(val)
							t[name] = key
						end
					else
						t[name] = conv_func(val)
					end
				end
				index = index + 1
			else
				table.insert(t, val)
			end
			fieldstart = string.find(s, ',', i) + 1
		else -- unquoted; find next comma
			local nexti = string.find(s, ',', fieldstart)
			val = string.sub(s, fieldstart,nexti-1)
			if head then
				name, conv_func, isKey = getFieldInfo(head[index])
				if conv_func then
					if isKey then
						if key then
							key1 = conv_func(val)
							t[name] = key1
						else
							key = conv_func(val)
							t[name] = key
						end
					else
						t[name] = conv_func(val)
					end
				end
				index = index + 1
			else
				table.insert(t, val)
			end
			fieldstart = nexti + 1
		end
	until fieldstart > string.len(s)
	return t, key, key1
end

function readCSV(file)
	print('read the configuration file [' .. file .. ']')
	return readCSVEx(file, 2, true)
end

function readCSVEx(file, start_line, table_head)
	local fp = assert(io.open (file))
	local csv = {}
	local count = 0
	local first_line = true
	local head = nil
	local row = nil
	local key = nil
	local key1 = nil
	local data = nil

	local lastLine = ""
	local quote = false

	local function checkCompleteLine(line)
		for i = 1, #line do
			if string.byte(line, i) == string.byte('"') then
				quote = not quote
			end
		end
		if quote then
			lastLine = lastLine .. line .. "\n"
			return false, nil
		else
			local completeLine = lastLine .. line
			lastLine  = ""
			return true, completeLine
		end
	end

	local complete
	local line

	for l in fp:lines() do
		complete, line = checkCompleteLine(l)
		if complete then
			count = count + 1
			if count >= start_line then
				if table_head then
					if first_line then
						head = fromCSV(line)
					else
						data, key, key1 = fromCSV(line, head)
						if key then
							if key1 then
								if not csv[key] then
									csv[key] = {}
								end
								csv[key][key1] = data
							else
								csv[key] = data
							end
						else
							csv[#csv+1] = data
						end
					end
				else
					csv[#csv+1] = fromCSV(line)
				end
				first_line = false
			end
		end
	end
	return csv
end

local function splitLine(s, head)
	s = s .. ',' -- ending comma
	local sl
	local cl
	local fieldstart = 1
	local index = 1
	local val = nil

	repeat
		-- next field is quoted? (start with `"'?)
		if string.find(s, '^"', fieldstart) then
			local a, c
			local i = fieldstart
			repeat
				-- find closing quote
				a, i, c = string.find(s, '"("?)', i+1)
			until c ~= '"' -- quote not followed by quote?

			if not i then error('unmatched "') end
			local f = string.sub(s, fieldstart+1, i-1)
			val = string.gsub(f, '""', '"')

			local location = getFieldLocation(head[index])
			if location == 1 then
				if cl then
					cl = cl .. ',"' .. f .. '"'
				else
					cl = '"' .. f .. '"'
				end
			elseif location == 2 then
				if sl then
					sl = sl .. ',"' .. f .. '"'
				else
					sl = '"' .. f .. '"'
				end
			elseif location == 3 then
				if cl then
					cl = cl .. ',"' .. f .. '"'
				else
					cl = '"' .. f .. '"'
				end
				if sl then
					sl = sl .. ',"' .. f .. '"'
				else
					sl = '"' .. f .. '"'
				end
			end

			index = index + 1
			fieldstart = string.find(s, ',', i) + 1

		else -- unquoted; find next comma
			local nexti = string.find(s, ',', fieldstart)
			val = string.sub(s, fieldstart,nexti-1)
			local location = getFieldLocation(head[index])
			if location == 1 then
				if cl then
					cl = cl .. "," .. val
				else
					cl = val
				end
			elseif location == 2 then
				if sl then
					sl = sl .. "," .. val
				else
					sl = val
				end
			elseif location == 3 then
				if cl then
					cl = cl .. "," .. val
				else
					cl = val
				end
				if sl then
					sl = sl .. "," .. val
				else
					sl = val
				end
			end
			index = index + 1
			fieldstart = nexti + 1
		end
	until fieldstart > string.len(s)

	return cl, sl

end


function splitCSV(file, serverfile, clientfile)
	return splitCSVEx(file, 2, true, serverfile, clientfile)
end

function splitCSVEx(file, start_line, table_head, serverfile, clientfile)
	local fp = assert(io.open (file))
	local sfp = assert(io.open (serverfile, "w+"))
	local cfp = assert(io.open (clientfile, "w+"))
	local sf_is_empty = true
	local cf_is_empty = true
	local count = 0
	local first_line = true
	local head = nil
	local row = nil
	local key = nil
	local key1 = nil
	local data = nil

	local lastLine = ""
	local quote = false

	local function checkCompleteLine(line)
		for i = 1, #line do
			if string.byte(line, i) == string.byte('"') then
				quote = not quote
			end
		end
		if quote then
			lastLine = lastLine .. line .. "\n"
			return false, nil
		else
			local completeLine = lastLine .. line
			lastLine  = ""
			return true, completeLine
		end
	end

	local complete
	local line

	for l in fp:lines() do
		complete, line = checkCompleteLine(l)
		if complete then
			count = count + 1
			if count >= start_line then
				if table_head then
					if first_line then
						head = fromCSV(line)
					end

					local cl
					local sl
					cl, sl = splitLine(line, head)
					if cl then
						cfp:write(cl .. "\n")
						if first_line then
							cfp:write(cl .. "\n")
						end
						cf_is_empty = false
					end
					if sl then
						sfp:write(sl .. "\n")
						if first_line then
							sfp:write(sl .. "\n")
						end
						sf_is_empty = false
					end
				end
				first_line = false
			end
		end
	end

	fp:close()
	cfp:close()
	sfp:close()
	if cf_is_empty then
		os.remove(clientfile)
	end
	if sf_is_empty then
		os.remove(serverfile)
	end
end
