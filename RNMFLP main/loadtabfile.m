function [data,rows,cols]=loadtabfile(filename)
	% Load a table in the format of the drug-target interaction data provided by Yamanishi et al.
	%
	% This is a numeric table with row and column headings
	
	f = fopen(filename,'rt');
	
	line = fgetl(f);
	cols = textscan(line,'%s');
	cols = cols{1,1}; % first column for row labels
	
	rows = cell(0);
	data = zeros(0);
	while ~feof(f)
		line = fgetl(f);
		if numel(line) <= 1, break; end
		pos = find(line==9,1);
		rows = cat(1,rows,{line(1:pos-1)});
		data = cat(1,data,sscanf(line(pos:end),'%f')');
	end
	
	fclose(f);
end