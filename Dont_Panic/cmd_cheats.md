`grep` a Unix command used to search files for the occurrence of a string of characters that matches a specified pattern.

ex:

`ifconfig | grep inet`


the pipe | means you’re restricting the search to whatever comes before it ( inet). S

Sidenote: that command displays all your IP config info, but if you just want to see your internal IP address, here's a better way: 
`> ipconfig getifaddr en0`  … which will output something like
`192.168.1.12`