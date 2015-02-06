# parcels-sfr.Rmd
# main program to create files
# WORKING/parcels-sfr.RData: all features of single-family residential parcels
# WORKING/parcels-sfr-counts.csv: record counts

# File layout is in 2580...
# NOTE: the features of parcels are in the taxroll files

source('Directory.R')
source('InitializeR.R')
source('LUSEI.R')
source('Printf.R')
source('ReadRawParcels.R')

# set the control variables
Control <- function() {
    # return list of control variables
    me <-'parcels-sfr'
    
    log <- Directory('log')
    input <- Directory('input')
    working <- Directory('working')
    
    control <- list(
         path.out.log = paste0(log, me, '.log')
        ,path.out.deeds = paste0(working, me, '.RData')
        ,path.out.counts = paste0(working, me, '-counts.csv')
        ,path.to.input.directory = input
        ,testing = FALSE
        )
    control
}
Main <- function(control) {
    #cat('start Main\n'); browser()
    
    # write control variables
    InitializeR(duplex.output.to = control$path.out.log)
    str(control)

    # read all the parcels
    all <- ReadRawParcels( nrows = if (control$testing) 1000 else -1
                          ,path.to.input.directory = control$path.to.input.directory
                          ,verbose = TRUE
                          )

    # keep only info about the parcels, not about the deeds
    #cat('drop deeds info\n'); browser()

    keeps <- c(  "APN.UNFORMATTED"                              
               , "APN.SEQUENCE.NUMBER"                          
               , "APN.FORMATTED"                                
               , "MAP.REFERENCE.1"                              
               , "MAP.REFERENCE.2"                              
               , "CENSUS.TRACT"                                 
               , "CENSUS.BLOCK.GROUP"                           
               , "CENSUS.BLOCK"                                 
               , "CENSUS.BLOCK.SUFFIX"                          
               , "ZONING"                                       
               , "BLOCK.NUMBER"                                 
               , "LOT.NUMBER"                                   
               , "RANGE"                                        
               , "TOWNSHIP"                                     
               , "SECTION"                                      
               , "QUARTER.SECTION"                              
               , "THOMAS.BROS.MAP.NUMBER"                       
               , "FLOOD.ZONE.COMMUNITY.PANEL.ID"                
               , "CENTROID.CODE"                                
               , "HOMESTEAD.EXEMPT"                             
               , "ABSENTEE.INDICATOR.CODE"                      
               , "TAX.CODE.AREA"                                
               , "UNIVERSAL.LAND.USE.CODE"                      
               , "COUNTY.LAND.USE.1"                            
               , "COUNTY.LAND.USE.2"                            
               , "PROPERTY.INDICATOR.CODE"                      
               , "MUNICIPALITY.NAME"                            
               , "VIEW"                                         
               , "LOCATION.INFLUENCE.CODE"                      
               , "NUMBER.OF.BUILDINGS"                          
               , "SUBDIVISION.TRACT.NUMBER"                     
               , "SUBDIVISION.PLAT.BOOK"                        
               , "SUBDIVISION.PLAT.PAGE"                        
               , "SUBDIVISION.NAME"                             
               , "PROPERTY.ADDRESS.INDICATOR.CODE"              
               , "PROPERTY.HOUSE.NUMBER.PREFIX"                 
               , "PROPERTY.HOUSE.NUMBER"                        
               , "PROPERTY.HOUSE.NUMBER.SUFFIX"                 
               , "PROPERTY.DIRECTION"                           
               , "PROPERTY.STREET.NAME"                         
               , "PROPERTY.MODE"                                
               , "PROPERTY.QUADRANT"                            
               , "PROPERTY.APARTMENT.UNIT.NUMBER"               
               , "PROPERTY.CITY"                                
               , "PROPERTY.STATE"                               
               , "PROPERTY.ZIPCODE"                             
               , "PROPERTY.CARRIER.ROUTE"                       
               , "PROPERTY.MATCH.CODE"                          
               , "OWNER.CORPORATE.INDICATOR.FLAG"               
               , "MAIL.HOUSE.NUMBER.PREFIX"                     
               , "MAIL.HOUSE.NUMBER"                            
               , "MAIL.HOUSE.NUMBER.SUFFIX"                     
               , "MAIL.DIRECTION"                               
               , "MAIL.STREET.NAME"                             
               , "MAIL.MODE"                                    
               , "MAIL.QUADRANT"                                
               , "MAIL.APARTMENT.UNIT.NUMBER"                   
               , "MAIL.CITY"                                    
               , "MAIL.STATE"                                   
               , "MAIL.ZIPCODE"                                 
               , "MAIL.CARRIER.ROUTE"                           
               , "MAIL.MATCH.CODE"                              
               , "MAIL.OPT.OUT.FLAG"                            
               , "TOTAL.VALUE.CALCULATED"                       
               , "LAND.VALUE.CALCULATED"                        
               , "IMPROVEMENT.VALUE.CALCULATED"                 
               , "TOTAL.VALUE.CALCULATED.INDICATOR.FLAG"        
               , "LAND.VALUE.CALCULATED.INDICATOR.FLAG"         
               , "IMPROVEMENT.VALUE.CALCULATED.INDICATOR.FLAG"  
               , "ASSD.TOTAL.VALUE"                             
               , "ASSD.LAND.VALUE"                              
               , "ASSD.IMPROVEMENT.VALUE"                       
               , "MKT.TOTAL.VALUE"                              
               , "MKT.LAND.VALUE"                               
               , "MKT.IMPROVEMENT.VALUE"                        
               , "APPR.TOTAL.VALUE"                             
               , "APPR.LAND.VALUE"                              
               , "APPR.IMPROVEMENT.VALUE"                       
               , "TAX.AMOUNT"                                   
               , "TAX.YEAR"                                     
               , "BATCH.ID"                                     
               , "BATCH.SEQ"                                    
               # "DOCUMENT.YEAR"                                
               # "SALES.DOCUMENT.TYPE.CODE"                     
               # "RECORDING.DATE"                               
               # "SALE.DATE"                                    
               # "SALE.AMOUNT"                                  
               # "SALE.CODE"                                    
               # "SALES.TRANSACTION.TYPE.CODE"                  
               # "MULTI.APN.FLAG.CODE"                          
               # "MULTI.APN.COUNT"                              
               # "RESIDENTIAL.MODEL.INDICATOR.FLAG"             
               # "X1ST.MORTGAGE.AMOUNT"                         
               # "X1ST.MORTGAGE.DATE"                           
               # "X1ST.MORTGAGE.LOAN.TYPE.CODE"                 
               # "X1ST.MORTGAGE.DEED.TYPE.CODE"                 
               # "X1ST.MORTGAGE.TERM.CODE"                      
               # "X1ST.MORTGAGE.TERM"                           
               # "X1ST.MORTGAGE.DUE.DATE"                       
               # "X1ST.MORTAGE.ASSUMPTION.AMOUNT"               
               # "X2ND.MORTGAGE.AMOUNT"                         
               # "X2ND.MORTGAGE.LOAN.TYPE.CODE"                 
               # "X2ND.DEED.TYPE.CODE"                          
               # "PRIOR.SALE.TRANSACTION.ID"                    
               # "PRIOR.SALE.DOCUMENT.YEAR"                     
               # "PRIOR.SALE.DOCUMENT.NUMBER"                   
               # "PRIOR.SALE.BOOK.PAGE"                         
               # "PRIOR.SALE.DOCUMENT.TYPE.CODE"                
               # "PRIOR.SALE.RECORDING.DATE"                    
               # "PRIOR.SALE.DATE"                              
               # "PRIOR.SALE.AMOUNT"                            
               # "PRIOR.SALE.CODE"                              
               # "PRIOR.SALE.TRANSACTION.TYPE.CODE"             
               # "PRIOR.SALE.MULTI.APN.FLAG.CODE"               
               # "PRIOR.SALE.MULTI.APN.COUNT"                   
               # "PRIOR.SALE.MORTGAGE.AMOUNT"                   
               # "PRIOR.SALE.DEED.TYPE.CODE"                    
               , "FRONT.FOOTAGE"                                
               , "DEPTH.FOOTAGE"                                
               , "ACRES"                                        
               , "LAND.SQUARE.FOOTAGE"                          
               , "LOT.AREA"                                     
               , "UNIVERSAL.BUILDING.SQUARE.FEET"               
               , "UNIVERSAL.BUILDING.SQUARE.FEET.INDICATOR.CODE"
               , "BUILDING.SQUARE.FEET"                         
               , "LIVING.SQUARE.FEET"                           
               , "GROUND.FLOOR.SQUARE.FEET"                     
               , "GROSS.SQUARE.FEET"                            
               , "ADJUSTED.GROSS.SQUARE.FEET"                   
               , "BASEMENT.SQUARE.FEET"                         
               , "GARAGE.PARKING.SQUARE.FEET"                   
               , "YEAR.BUILT"                                   
               , "EFFECTIVE.YEAR.BUILT"                         
               , "BEDROOMS"                                     
               , "TOTAL.ROOMS"                                  
               , "TOTAL.BATHS.CALCULATED"                       
               , "TOTAL.BATHS"                                  
               , "FULL.BATHS"                                   
               , "HALF.BATHS"                                   
               , "X1QTR.BATHS"                                  
               , "X3QTR.BATHS"                                  
               , "BATH.FIXTURES"                                
               , "AIR.CONDITIONING.CODE"                        
               , "BASEMENT.FINISH.CODE"                         
               , "BLDG.CODE"                                    
               , "BLDG.IMPROVEMENT.CODE"                        
               , "CONDITION.CODE"                               
               , "CONSTRUCTION.TYPE.CODE"                       
               , "EXTERIOR.WALLS.CODE"                          
               , "FIREPLACE.INDICATOR.FLAG"                     
               , "FIREPLACE.NUMBER"                             
               , "FIREPLACE.TYPE.CODE"                          
               , "FOUNDATION.CODE"                              
               , "FLOOR.CODE"                                   
               , "FRAME.CODE"                                   
               , "GARAGE.CODE"                                  
               , "HEATING.CODE"                                 
               , "MOBILE.HOME.INDICATOR.FLAG"                   
               , "PARKING.SPACES"                               
               , "PARKING.TYPE.CODE"                            
               , "POOL.FLAG"                                    
               , "POOL.CODE"                                    
               , "QUALITY.CODE"                                 
               , "ROOF.COVER.CODE"                              
               , "ROOF.TYPE.CODE"                               
               , "STORIES.CODE"                                 
               , "STORIES.NUMBER"                               
               , "STYLE.CODE"                                   
               , "UNITS.NUMBER"                                 
               , "ELECTRIC.ENERGY.CODE"                         
               , "FUEL.CODE"                                    
               , "SEWER.CODE"                                   
               , "WATER.CODE"                                   
               , "LEGAL.1"                                      
               , "LEGAL.2"                                      
               , "LEGAL.3"  
               )

    # make sure all of the to-be-kept fields are present
    lapply( keeps
           ,function(one.keep) {
               cat(one.keep, '\n')
               all[one.keep]  # this will fail of one.keep is not a column
           }
           )
               
    # In addition, retain only observations coded as single-family residential
    is.sfr <- LUSEI(all$UNIVERSAL.LAND.USE.CODE, 'sfr')
    parcels.sfr <- all[is.sfr, keeps]

    # count records
    nrow.all <- nrow(all)
    nrow.sfr <- nrow(parcels.sfr)
    
    Printf('Read %d deeds\n', nrow.all)
    Printf('Retained %d as single-family residential\n', nrow.sfr)
    str(parcels.sfr)
    
    # Write RData
    cat('writing output\n')
    info <- list( nrow.all = nrow.all
                 ,nrow.sfr = nrow.sfr
    )
    save(parcels.sfr, info, control, file = control$path.out.deeds)
    counts <- data.frame( stringsAsFactors = FALSE
                        ,file_name = c('raw', 'sfr')
                        ,record_count = c(nrow.all, nrow.sfr)
                        )
    write.csv(counts, file = control$path.out.counts)


    # write control variables
    str(control)
    if (control$testing)
        cat('DISCARD OUTPUT: TESTING\n')
    
}

Main(Control())
cat('done\n')
